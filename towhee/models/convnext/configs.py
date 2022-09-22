# Model weights and configs are from https://github.com/facebookresearch/ConvNeXt
#
# Modifications & additions by Copyright 2021 Zilliz. All rights reserved.
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
        'depths': (3, 3, 27, 3),
        'dims': (128, 256, 512, 1024),
        'num_classes': 1000
    }
    if model_name is None:
        pass
    elif model_name == 'convnext_tiny_1k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
            depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
        )
    elif model_name == 'convnext_tiny_22k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224_ema.pth',
            depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), num_classes=21841,
        )
    elif model_name == 'convnext_small_1k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth',
             dims=(96, 192, 384, 768),
        )
    elif model_name == 'convnext_small_22k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224_ema.pth',
            dims=(96, 192, 384, 768), num_classes=21841,
        )
    elif model_name == 'convnext_base_1k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth',
        )
    elif model_name == 'convnext_base_22k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224_ema.pth',
            num_classes=21841,
        )
    elif model_name == 'convnext_large_1k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth',
            dims=(192, 384, 768, 1536),
        )
    elif model_name == 'convnext_large_22k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224_ema.pth',
            dims=(192, 384, 768, 1536), num_classes=21841,
        )
    elif model_name == 'convnext_xlarge_22k':
        configs.update(
            url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224_ema.pth',
            dims=(256, 512, 1024, 2048), num_classes=21841,
        )
    else:
        raise ValueError(f'Invalid model name: {model_name}.'
                         f'Expected model name from [convnext_tiny_1k, convnext_tiny_22k, '
                         f'convnext_small_1k, convnext_small_22k, '
                         f'convnext_base_1k, convnext_base_22k, '
                         f'convnext_large_1k, convnext_large_22k, convnext_xlarge_22k]')
    return configs
