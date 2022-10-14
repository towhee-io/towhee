# Configuration of RepLKNet model in the paper:
#       "Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs"
#       (https://arxiv.org/abs/2203.06717)
#
# Inspired by the original code from https://github.com/DingXiaoH/RepLKNet-pytorch
#
# All additions & modifications by Copyright 2021 Zilliz. All rights reserved.
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

def get_configs(model_name: str = None):
    if model_name is None:
        configs = {
            'large_kernel_sizes': [31, 29, 27, 13],
            'layers': [2, 2, 18, 2],
            'channels': [128, 256, 512, 1024],
            'small_kernel': 5,
            'dw_ratio': 1,
            'ffn_ratio': 4,
            'in_channels': 3,
            'drop_rate': 0.3,
            'num_classes': 1000,
            'out_indices': None,
            'small_kernel_merged': False,
            'norm_intermediate_features': False,
            'deep_fuse': True
        }
    else:
        # todo: add configs for different model names
        raise AttributeError(f'Invalid model name "{model_name}".')
    return configs
