# Copyright 2021 Zilliz. All rights reserved.
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

from typing import Union

import torch.cuda
from torch import nn

from towhee.models import clip
from .action_clip_utils import map_state_dict, get_configs
from .text_prompt import text_prompt
from .visual_prompt import VisualPrompt


class ActionClip:
    """
    ActionClip model.

    Args:
        - clip_model (`nn.Module`):
            CLIP model
        - visual_prompt (`nn.Module`):
            Visual prompt.
        - text_prompt_flat (`bool`):
            Flag to control whether to use text prompt.
    """
    def __init__(
            self,
            clip_model: nn.Module,
            visual_prompt: nn.Module = None,
            text_prompt_flag: bool = True
            ):
        super().__init__()
        self.clip_model = clip_model
        self.visual_prompt = visual_prompt
        self.text_prompt_flag = text_prompt_flag

    def encode_video(self, video):
        b, t, c, h, w = video.size()
        frames = video.view(-1, c, h, w)
        if hasattr(self.clip_model, "encode_image"):
            frame_features = self.clip_model.encode_image(frames)
        else:
            frame_features = self.clip_model(frames)
        image_features = frame_features.view(b, t, -1)
        if self.visual_prompt:
            visual_features = self.visual_prompt(image_features)
        else:
            visual_features = image_features
        return visual_features

    def encode_text(self, text: list):
        if self.text_prompt_flag:
            text, _, _ = text_prompt(text)
        if hasattr(self.clip_model, "encode_text"):
            text_features = self.clip_model.encode_text(text)
        else:
            text_features = self.clip_model(text)
        return text_features


def create_model(
        clip_model: Union[str, nn.Module],
        pretrained: bool = False,
        jit: bool = False,
        device: str = None,
        checkpoints: dict = None,
        cfg: dict = None
        ) -> ActionClip:
    """
    Create action clip model
    Args:
        clip_model (`Union[str, nn.Module]`):
            A clip model name or a model instance.
        pretrained (`bool`):
            Whether model is pretrained.
        jit (`bool`):
            Whether returned one is a jit model, only useful when `pretrained` is True.
        device (`str`):
            Cuda or gpu.
        checkpoints (`dict`):
            Checkpoints state infos, only useful when `pretrained` is True.
        cfg (`dict`):
            Other model configs.

    Returns:
        (`ActionClip`)
            ActionClip model instance.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(clip_model, str):
        configs = get_configs(model_name=clip_model)
        clip_model = clip.create_model(model_name=clip_model, pretrained=pretrained, jit=jit, device=device)
    else:
        configs = get_configs()
    if cfg is not None:
        configs.update(**cfg)
    visual_prompt_type = configs["visual_prompt_type"]
    num_frames = configs["num_frames"]
    visual_prompt = VisualPrompt(visual_prompt_type, clip_model.state_dict(), num_frames).to(device)
    if pretrained and checkpoints:
        if "clip_state_dict" in checkpoints:
            clip_state_dict = map_state_dict(checkpoints, "clip_state_dict")
            clip_model.load_state_dict(clip_state_dict)
        if "visual_state_dict" in checkpoints:
            visual_state_dict = map_state_dict(checkpoints, "visual_state_dict")
            visual_prompt.load_state_dict(visual_state_dict)
    model = ActionClip(clip_model=clip_model, visual_prompt=visual_prompt)
    return model
