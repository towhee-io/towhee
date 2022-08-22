# Original pytorch implementation by:
# 'Frozen in Time: A Joint Image and Video Encoder for End-to-End Retrieval'
#       - https://arxiv.org/abs/2104.00650
# Original code by / Copyright 2021, Max Bain.
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
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo
from transformers import AutoModel, AutoConfig
from towhee.models.frozen_in_time.frozen_video_transformer import SpaceTimeTransformer
from towhee.models import vit
from towhee.models.frozen_in_time.frozen_utils import sim_matrix, state_dict_data_parallel_fix, \
    remove_bridge_module_state_dic
import logging
# if load frozen pretrained model , must import this package
# beacause this pretrained model is base on  https://github.com/victoresque/pytorch-template
# from towhee.models.frozen_in_time.parse_config import ConfigParser
# # this line for the pass pylint
# ConfigParser()


class FrozenInTime(nn.Module):
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
        attention_style (str):
            default is "frozen_in_time", if is "bridge_former", then is the BridgeFormer model
        projection_dim (int):
            the output dim
        video_pretrained_model (str):
            default is "vit_base_16x224", if is_pretrained is false, then must load the pretrained vit parameters
        video_is_load_pretrained (bool):
            if true, then will load vit pretrained weight
        video_model_type (str) :
            default is "SpaceTimeTransformer", the frozen model name, no need to change
        text_pretrained_model (str):
            default is "distilbert-base-uncased", the pretrained model for text encoder
        text_is_load_pretrained (bool):
            if false, then will not load pretrained model for text encoder
        projection (str):
            a layer
        load_temporal_fix (str):
            if is_pretrained is true, this will use in function "_inflate_positional_embeds"
        device (str):
            "cpu" or "cuda"
        pretrained_url (str):
            if is_pretrained is true and weights_path is None, will load frozen pretrained model from this url

    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_frames: int = 4,
                 is_pretrained: bool = False,
                 weights_path: str = '',
                 attention_style: str = 'frozen_in_time',
                 projection_dim: int = 256,
                 video_pretrained_model: str = 'vit_base_16x224',
                 video_is_load_pretrained: bool = False,
                 video_model_type: str = 'SpaceTimeTransformer',
                 text_pretrained_model: str = 'distilbert-base-uncased',
                 text_is_load_pretrained: bool = False,
                 projection: str = 'minimal',
                 load_temporal_fix: str = 'zeros',
                 device: str = 'cpu',
                 pretrained_url: str = 'https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid2m-4f_stformer_b_16_224.pth.tar'
                 ):
        super().__init__()

        self.text_pretrained_model = text_pretrained_model
        self.video_model_type = video_model_type
        self.num_frames = num_frames
        self.load_temporal_fix = load_temporal_fix
        self.attention_style = attention_style
        if text_is_load_pretrained:
            # load text pretrained model
            self.text_model = AutoModel.from_pretrained(self.text_pretrained_model)
        else:
            self.config = AutoConfig.from_pretrained(self.text_pretrained_model)
            self.text_model = AutoModel.from_config(config=self.config)

        self.text_model.train()

        # load video pretrained model
        if video_model_type == 'SpaceTimeTransformer':
            model = SpaceTimeTransformer(img_size=img_size,
                                         patch_size=patch_size,
                                         in_chans=in_chans,
                                         num_frames=num_frames,
                                         time_init='zeros',
                                         attention_style=attention_style)
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
        else:
            raise NotImplementedError(f'{video_model_type} not implemented')
        if self.attention_style == 'bridge_former':
            self.video_model.fc = nn.Identity()
        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        if self.attention_style == 'bridge_former':
            self.text_proj = txt_proj
        else:
            self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if is_pretrained:

            if weights_path not in ['', None]:
                checkpoint = torch.load(weights_path, map_location=device)
                state_dict = checkpoint['state_dict']
            else:
                if self.attention_style == 'bridge_former':
                    raise NotImplementedError(f'{self.attention_style} not have the weights url, '
                                              f'need give the weights path')

                checkpoint = model_zoo.load_url(pretrained_url, map_location=torch.device(device))
                state_dict = checkpoint['state_dict']

            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            if self.attention_style == 'bridge_former':
                new_state_dict = remove_bridge_module_state_dic(new_state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)

    def forward(self, text, video, return_embeds=True):
        # video shape (batch x frames x channels x height x width) type is tensor
        # text shape [bs, seq_len,embedding_dim] type is dic {'input_ids': tensor, 'attention_mask': tensor}
        text_data = text
        video_data = video
        # [bs,projection_dim]
        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data):
        text_pretrained_model = os.path.split(self.text_pretrained_model)[1]
        # print(text_pretrained_model)
        if text_pretrained_model.find('distilbert') != -1:
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        elif text_pretrained_model.find('bert') != -1:
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        else:
            raise NotImplementedError
        if self.attention_style == 'bridge_former':
            text_embeddings = self.text_proj(text_embeddings)
        else:
            text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

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


# if __name__ == "__main__":
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     text_pretrained_model_or_path = '/Users/zilliz/PycharmProjects/pretrain/distilbert-base-uncased'
#     frozen_in_time_pretrained_model_path = None
#     path = '/Users/zilliz/PycharmProjects/pretrain/BridgeFormer/MCQ.pth'
#     # frozen-in-time frozen_in_time
#     model = FrozenInTime(
#                          attention_style='bridge_former',
#                          is_pretrained=True,
#                          weights_path=path,
#                          projection_dim=256,
#                          text_pretrained_model=text_pretrained_model_or_path,
#                          video_pretrained_model='vit_base_16x224',
#                          video_is_load_pretrained=False,
#                          video_model_type='SpaceTimeTransformer',
#                          text_is_load_pretrained=False,
#                          device=device)
#
#     # (batch x channels x frames x height x width)
#     # dummy_video = torch.randn(1, 3, 4, 224, 224)
#     # img = origial_img.unsqueeze(2)
#     # dummy_video = torch.tensor(img.repeat(1, 1, 4, 1, 1))  # [1, 3, 4, 224, 224]
#     # (batch x channels x frames x height x width)
#     dummy_video = torch.randn(1, 4, 3, 224, 224)
#     # img = origial_img.unsqueeze(0)
#     # dummy_video = torch.tensor(img.repeat(1, 4, 1, 1, 1))  # [1, 4, 3, 224, 224]
#
#     # text = ['study as you']
#     # tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model_or_path, TOKENIZERS_PARALLELISM=False)
#     # dummy_text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
#     input_ids = torch.randint(1, 10, size=(1, 64))
#     attention_mask = torch.ones(1, 64, dtype=torch.int)
#     dummy_text = dict()
#     dummy_text['input_ids'] = input_ids
#     dummy_text['attention_mask'] = attention_mask
#     for key in dummy_text.keys():
#         print(key, dummy_text[key].shape)
#
#     print(dummy_video.shape)
#     text_embeddings, video_embeddings = model(text=dummy_text, video=dummy_video, return_embeds=True)
#     assert text_embeddings.shape == (1, 256)
#     assert video_embeddings.shape == (1, 256)
#     text_with_video_sim = model(text=dummy_text, video=dummy_video, return_embeds=False)
#     print(text_with_video_sim)
#     assert text_with_video_sim.shape == (1,1)
