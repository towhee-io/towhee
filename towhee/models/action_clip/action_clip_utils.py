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

from collections import OrderedDict

import torch


def map_state_dict(checkpoint, state_dict_name: str = None):
    if state_dict_name and state_dict_name in checkpoint:
        old_state_dict = checkpoint[state_dict_name]
    else:
        old_state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v
    return new_state_dict


def get_similarity(text_features: torch.Tensor, visual_features: torch.Tensor, num_text_augs: int,
                   norm: bool = False) -> torch.Tensor:
    if norm:
        text_features /= text_features.norm(dim=-1, keepdim=True)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ visual_features.T)
    similarity = similarity.view(visual_features.size(0), num_text_augs, -1).softmax(dim=-1)
    similarity = similarity.mean(dim=1, keepdim=False)
    return similarity


def get_configs(model_name: str = None, **kwargs):
    configs = dict(
        visual_prompt_type='Transf',
        num_frames=8
    )
    configs.update(**kwargs)
    if model_name == 'clip_vit_b16':
        configs.update(
            num_frames=8
        )
    elif model_name == 'clip_vit_b32':
        configs.update(
            num_frames=8
        )
    return configs
