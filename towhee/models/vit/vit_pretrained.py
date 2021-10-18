# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
A collection of urls for open sourced models.
"""

from .vit_configs import (
    get_b16_config,
    get_b32_config,
    get_l16_config,
    get_l32_config,
    drop_head_variant
)


PRETRAINED_MODELS = {
    'B_16': {
      'config': get_b16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth'
    },
    'B_32': {
      'config': get_b32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth'
    },
    'L_16': {
      'config': get_l16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': None
    },
    'L_32': {
      'config': get_l32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth'
    },
    'B_16_imagenet1k': {
      'config': drop_head_variant(get_b16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'
    },
    'B_32_imagenet1k': {
      'config': drop_head_variant(get_b32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth'
    },
    'L_16_imagenet1k': {
      'config': drop_head_variant(get_l16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth'
    },
    'L_32_imagenet1k': {
      'config': drop_head_variant(get_l32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': 'https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth'
    },
}
