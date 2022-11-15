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
    if model_name is None:
        pass
    elif model_name == 'replk_31b_1k':
        configs.update(
            url='https://drive.google.com/uc?export=download&confirm=9_s_&id=1DslZ2voXZQR1QoFY9KnbsHAeF84hzS0s'
        )
    elif model_name == 'replk_31b_22k':
        configs.update(
            url='https://drive.google.com/uc?export=download&confirm=9_s_&id=1PYJiMszZYNrkZOeYwjccvxX8UHMALB7z',
            num_classes=21841
        )
    elif model_name == 'replk_31l_1k':
        configs.update(
            url='https://drive.google.com/uc?export=download&confirm=9_s_&id=1JYXoNHuRvC33QV1pmpzMTKEni1hpWfBl',
            channels=[192, 384, 768, 1536],
        )
    elif model_name == 'replk_31l_22k':
        configs.update(
            url='https://drive.google.com/uc?export=download&confirm=9_s_&id=16jcPsPwo5rko7ojWS9k_W-svHX-iFknY',
            channels=[192, 384, 768, 1536],
            num_classes=21841
        )
    elif model_name == 'replk_xl':
        configs.update(
            url='https://drive.google.com/uc?export=download&confirm=9_s_&id=1tPC60El34GntXByIRHb-z-Apm4Y5LX1T',
            large_kernel_sizes=[27, 27, 27, 13],
            channels=[256, 512, 1024, 2048],
            small_kernel=None, dw_ratio=1.5,
        )
    else:
        raise AttributeError(f'Invalid model name "{model_name}".')
    return configs
