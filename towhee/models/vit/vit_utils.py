# Original pytorch implementation by:
# 'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
#       - https://arxiv.org/abs/2010.11929
# 'How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers'
#       - https://arxiv.org/abs/2106.10270
#
# Model weights are provided by corresponding links and credit to original authors.
# Original code by / Copyright 2020, Ross Wightman.
# Modifications & additions by / Copyright 2021 Zilliz. All rights reserved.
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
from torch.utils import model_zoo


def load_pretrained_model(model, configs: dict, weights_path: str = None):
    if weights_path:
        state_dict = torch.load(weights_path)
    else:
        state_dict = model_zoo.load_url(configs['url'], map_location=torch.device('cpu'))
    res = model.load_state_dict(state_dict, strict=False)
    return res

def base_configs():
    return dict(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_ratio=0,
    )


def get_configs(model_name):
    if model_name == 'vit_base_patch16_224':
        configs = base_configs()
        configs.update(dict(
            url='https://github.com/jaelgu/towhee_test/releases/download/v1.0.0/vit_base_patch16_224.pth'
        ))
        return configs
