# Configs of Shunted Transform in the paper:
#   [Shunted Self-Attention via Multi-Scale Token Aggregation](https://arxiv.org/abs/2111.15193)
#
# Configs & model weights are from https://github.com/OliverRensu/Shunted-Transformer
#
# Additions & modifications are protected by Copyright 2021 Zilliz. All rights reserved.
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

from functools import partial
from torch import nn
from collections import OrderedDict


def get_configs(model_name: str = None):
    configs = {
        'img_size': 224,
        'num_classes': 1000,
        'embed_dims': [64, 128, 256, 512],
        'num_heads': [2, 4, 8, 16],
        'mlp_ratios': [8, 8, 4, 4],
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': [3, 4, 24, 2],
        'sr_ratios': [8, 4, 2, 1],
        'num_stages': 4,
        'num_conv': 2
    }

    if model_name is None:
        pass
    elif model_name == 'shunted_t':
        configs.update(
            depths=[1, 2, 4, 1], num_conv=0
        )
    elif model_name == 'shunted_s':
        configs.update(
            depths=[2, 4, 12, 1], num_conv=1
        )
    elif model_name == 'shunted_b':
        pass
    else:
        raise ValueError(f'Invalid model name: {model_name}.')
    return configs


def convert_state_dict(state_dict: OrderedDict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace('dwconv.dwconv', 'dwconv')
        new_state_dict[k] = v
    return new_state_dict
