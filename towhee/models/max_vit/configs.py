# Original pytorch implementation by:
# 'MaxViT: Multi-Axis Vision Transformer'
#       - https://arxiv.org/pdf/2204.01697.pdf
# Original code by / Copyright 2021, Christoph Reich.
# Modifications & additions by / Copyright 2022 Zilliz. All rights reserved.
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


def get_configs(model_name):
    configs = {
        'max_vit_tiny': {'depths': (2, 2, 5, 2),
                         'channels': (64, 128, 256, 512),
                         'embed_dim': 64,
                         },
        'max_vit_small': {'depths': (2, 2, 5, 2),
                          'channels': (96, 192, 384, 768),
                          'embed_dim': 64,
                          },
        'max_vit_base': {'depths': (2, 6, 14, 2),
                         'channels': (96, 192, 384, 768),
                         'embed_dim': 64,
                         },
        'max_vit_large': {'depths': (2, 6, 14, 22),
                          'channels': (128, 256, 512, 1024),
                          'embed_dim': 128,
                          },
        'max_vit_xlarge': {'depths': (2, 6, 14, 22),
                           'channels': (192, 384, 768, 1536),
                           'embed_dim': 192,
                           },
    }
    return configs[model_name]


