# Original pytorch implementation by:
# 'Bridging Video-text Retrieval with Multiple Choice Questions'
#       - https://arxiv.org/pdf/2201.04850.pdf
# Original code by / Copyright 2022, Yuying Ge.
# Modifications & additions by / Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo
from transformers import AutoModel, AutoConfig
from towhee.models.bridgeformer.bridge_former_training_block import VideoBridgeFormer
from towhee.models import vit
from towhee.models.frozen_in_time.frozen_utils import state_dict_data_parallel_fix
import logging


class BridgeFormerTraining(nn.Module):
    """
    FrozenInTime model
    Args:
        img_size (int):
          the image or video height
        patch_size (int):
            patch height (equal to width)
        in_chans (int):
            number of image channel
        num_frames (int):
            maximum number of frames expected as input
        is_pretrained (bool):
            if true, then load the frozen pretrained model
        weights_path (str):
            the frozen pretrained model path, if is_pretrained is true, and this not None, then load from this path,
            otherwise, will load the frozen model from an url.

        projection_dim (int):
            the output dim
        video_pretrained_model (str):
            default is "vit_base_16x224", if is_pretrained is false, then must load the pretrained vit parameters
        video_is_load_pretrained (bool):
            if true, then will load vit pretrained weight

        text_pretrained_model (str):
            default is "distilbert-base-uncased", the pretrained model for text encoder
        text_is_load_pretrained (bool):
            if false, then will not load pretrained model for text encoder
        projection (str):
            a layer

    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_frames=4,
                 is_pretrained=False,
                 weights_path='',
                 projection_dim=256,
                 video_pretrained_model='vit_base_16x224',
                 video_is_load_pretrained=False,
                 text_pretrained_model='distilbert-base-uncased',
                 text_is_load_pretrained=False,
                 projection='minimal',
                 device='cpu',
                 pretrained_url=''
                 ):
        super().__init__()

        self.text_pretrained_model = text_pretrained_model
        self.num_frames = num_frames

        if text_is_load_pretrained:
            # load text pretrained model
            self.text_model = AutoModel.from_pretrained(self.text_pretrained_model)
        else:
            self.config = AutoConfig.from_pretrained(self.text_pretrained_model)
            self.config.output_hidden_states = True
            self.text_model = AutoModel.from_config(config=self.config)

        self.text_model.train()

        # load video pretrained model
        model = VideoBridgeFormer(img_size=img_size, patch_size=patch_size,
                                  in_chans=in_chans, num_frames=num_frames)

        model.head = nn.Identity()
        model.pre_logits = nn.Identity()
        ftr_dim = model.embed_dim
        # if not load pretrained model of frozen in time, then must load pretrained vit model to initial
        if (not is_pretrained) and video_is_load_pretrained:
            if video_pretrained_model == 'vit_base_16x224':
                vit_model = vit.create_model(model_name=video_pretrained_model, pretrained=True)
            else:
                raise NotImplementedError
            vit_checkpoint = vit_model.state_dict()
            model.load_state_dict(vit_checkpoint, strict=False)
        self.video_model = model
        self.video_model.fc = nn.Identity()
        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
            bridge_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )

        self.text_proj = txt_proj

        self.vid_proj = vid_proj
        self.bridge_proj = bridge_proj
        if is_pretrained:

            if weights_path not in ['', None]:

                checkpoint = torch.load(weights_path, map_location=device)
                state_dict = checkpoint['state_dict']
            else:
                checkpoint = model_zoo.load_url(pretrained_url, map_location=torch.device(device))
                state_dict = checkpoint['state_dict']

            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)

    def forward(self, text, answer_data, question_data, video):
        # video shape (batch x frames x channels x height x width) type is tensor
        # text、answer_data、question_data type is dic {'input_ids': tensor, 'attention_mask': tensor}
        text_data = text
        video_data = video
        # [bs,projection_dim]
        text_cls_embeddings, answer_cls_embeddings, question_embeddings, question_mask = \
            self.compute_text(text_data, answer_data, question_data)

        bridge_cls_embeddings, video_cls_embeddings = \
            self.compute_video(video_data, question_embeddings, question_mask)

        return text_cls_embeddings, answer_cls_embeddings, bridge_cls_embeddings, video_cls_embeddings

    def compute_text(self, text_data, answer_data, question_data):

        text_cls_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        answer_cls_embeddings = self.text_model(**answer_data).last_hidden_state[:, 0, :]

        text_cls_embeddings = self.text_proj(text_cls_embeddings)
        answer_cls_embeddings = self.text_proj(answer_cls_embeddings)

        question_embeddings = self.text_model(**question_data).hidden_states
        question_mask = question_data['attention_mask']

        return text_cls_embeddings, answer_cls_embeddings, question_embeddings, question_mask

    def compute_video(self, video_data, question_embeddings, question_mask):
        bridge_cls_embeddings, video_cls_embeddings = self.video_model(video_data, question_embeddings, question_mask)

        bridge_cls_embeddings = self.bridge_proj(bridge_cls_embeddings)
        video_cls_embeddings = self.vid_proj(video_cls_embeddings)

        return bridge_cls_embeddings, video_cls_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.num_frames
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    logging.info('### loaded %s model has MORE frames than current... '
                                 '### loading weights, filling in the extras via %s',
                                 self.video_model_type, self.load_temporal_fix)
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    logging.info('### loaded %s model has FEWER frames than current...'
                                 '### loading weights, filling in the extras via %s',
                                 self.video_model_type, self.load_temporal_fix)
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict

