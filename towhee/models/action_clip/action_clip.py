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

from torch import nn

# from .action_clip_utils import map_state_dict, get_similarity
from .text_prompt import text_prompt
from .visual_prompt import VisualPrompt


def encode_video(video, model: nn.Module, prompt: str = None):
    b, t, c, h, w = video.size()
    frames = video.view(-1, c, h, w)
    if hasattr(model, "encode_image"):
        frame_features = model.encode_image(frames)
    else:
        frame_features = model(frames)
    image_features = frame_features.view(b, t, -1)
    if prompt:
        fusion_model = VisualPrompt(prompt, model.state_dict(), t)
        visual_features = fusion_model(image_features)
    else:
        visual_features = image_features
    return visual_features


def encode_text(text, model: nn.Module, prompt: bool = True, return_long: bool = False):
    if prompt:
        text, num_text_augs, text_dict = text_prompt(text)
    if hasattr(model, "encode_text"):
        text_features = model.encode_text(text)
    else:
        text_features = model(text)
    if return_long:
        return text, num_text_augs, text_dict
    else:
        return text_features
