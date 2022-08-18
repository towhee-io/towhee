# Built on top of the original implementation at https://github.com/ArrowLuo/CLIP4Clip
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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
import logging

from typing import Union
from torch import nn
from towhee.models.clip4clip.until_module import PreTrainedModel, CrossEn
from towhee.models.clip import CLIP, get_configs

logger = logging.getLogger(__name__)


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self):
        super().__init__()
        self.clip = None


class CLIP4Clip(CLIP4ClipPreTrainedModel):
    """
    CLIP4Clip model from the paper: https://arxiv.org/abs/2104.08860
    Only meanP and 2d type reserved because of its performance.
    """

    def __init__(self, embed_dim: int,
                 image_resolution: int,
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.ignore_video_index = -1
        self.loose_type = True
        self.linear_patch = "2d"
        self.clip = CLIP(
            embed_dim=embed_dim, image_resolution=image_resolution,
            vision_layers=vision_layers, vision_width=vision_width, vision_patch_size=vision_patch_size,
            context_length=context_length, vocab_size=vocab_size, transformer_width=transformer_width,
            transformer_heads=transformer_heads, transformer_layers=transformer_layers,
            clip4clip=True
        ).float()
        self.sim_header = "meanP"
        self.loss_fct = CrossEn()
        self.apply(self.init_weights)

    def forward(self, input_ids, video, video_mask):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        # video_frame = bs * ts
        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, video, video_mask, shaped=True)
        if self.training:
            loss = 0.
            sim_matrix, _ = self.get_similarity_logits(sequence_output, visual_output, video_mask, shaped=True)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss
            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids, clip4clip=True).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, video, video_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        sequence_output = self.get_sequence_output(input_ids, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True)

        return sequence_output, visual_output

    def get_similarity_logits(self, sequence_output, visual_output, video_mask, shaped=False, ):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
        contrastive_direction = ()
        assert self.sim_header in ["meanP"]
        retrieve_logits = self._loose_similarity(sequence_output, visual_output, video_mask)
        return retrieve_logits, contrastive_direction

    def _loose_similarity(self, sequence_output, visual_output, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask, ):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out


def create_model(
        model_name: str = None,
        context_length: int = 77,
        pretrained: bool = False,
        weights_path: str = None,
        device: Union[str, torch.device] = None,
) -> CLIP4Clip:
    """
    Create a CLIP4Clip model.
    Args:
        model_name (`str`):
            Model name, `clip_vit_b16` or `clip_vit_b32`.
        context_length (`int`):
            Length of context sequence, default is 77, the same as these in CLIP.
        pretrained (`bool`):
            Whether pretained model.
        weights_path (`str`):
            Your local trained weights path. default is None.
        device (`str`):
            Model device, `cpu` or `cuda`.

    Returns:
        (`CLIP4Clip`):
            CLIP4Clip model.
    >>> from towhee.models import clip4clip
    >>> model = clip4clip.create_model(model_name="clip_vit_b32", context_length=32, pretrained=False, device='cpu')
    >>> model.__class__.__name__
    'CLIP4Clip'

    """
    configs = get_configs(model_name)
    if "multilingual_model" in configs:
        configs.pop("multilingual_model")
    if "url" in configs:
        configs.pop("url")
    configs["context_length"] = context_length
    model = CLIP4Clip(**configs)
    model.to(device)
    if pretrained and weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata  # pylint: disable=protected-access

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(   # pylint: disable=protected-access
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():   # pylint: disable=protected-access
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")

        if len(missing_keys) > 0:
            logger.info("Weights of %s not initialized from pretrained model: %s", model.__class__.__name__,
                        "\n   " + "\n   ".join(missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in %s: %s", model.__class__.__name__,
                        "\n   " + "\n   ".join(unexpected_keys))
        if len(error_msgs) > 0:
            logger.error("Weights from pretrained model cause errors in %s: %s", model.__class__.__name__,
                         "\n   " + "\n   ".join(error_msgs))

    if pretrained and weights_path is None:
        raise ValueError("weights_path is None")
    return model
