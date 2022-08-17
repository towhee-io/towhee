# Pytorch implementation of [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality]
# (https://arxiv.org/abs/2112.11081)
#
# Inspired by https://github.com/DingXiaoH/RepMLP
#
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

from towhee.utils.log import models_log


def get_configs(model_name: str = None):
    configs = {
        'channels': (96, 192, 384, 768),
        'hs': (56, 28, 14, 7),
        'ws': (56, 28, 14, 7),
        'num_blocks': (2, 2, 12, 2),
        'reparam_conv_k': (1, 3),
        'sharesets_nums': (1, 4, 32, 128),
    }
    if model_name == 'repmlp_b224':
        configs.update()
    elif model_name == 'repmlp_b256':
        configs.update(
            hs=(64, 32, 16, 8), ws=(64, 32, 16, 8)
        )
    elif model_name == 'repmlp_t224':
        configs.update(
            channels=(64, 128, 256, 512),
            num_blocks=(2, 2, 6, 2),
            sharesets_nums=(1, 4, 16, 128),
        )
    elif model_name == 'repmlp_t256':
        configs.update(
            channels=(64, 128, 256, 512),
            hs=(64, 32, 16, 8),
            ws=(64, 32, 16, 8),
            num_blocks=(2, 2, 6, 2),
            sharesets_nums=(1, 4, 16, 128)
        )
    elif model_name == 'repmlp_d256':
        configs.update(
            channels=(80, 160, 320, 640),
            hs=(64, 32, 16, 8), ws=(64, 32, 16, 8),
            num_blocks=(2, 2, 18, 2),
            sharesets_nums=(1, 4, 16, 128)
        )
    elif model_name == 'repmlp_l256':
        configs.update(
            channels=(96, 192, 384, 768),
            hs=(64, 32, 16, 8), ws=(64, 32, 16, 8),
            num_blocks=(2, 2, 18, 2),
            sharesets_nums=(1, 4, 32, 256),
        )
    else:
        models_log.warning('Model name "%s" is not supported.', model_name)
    return configs
