# Built on top of the original implementation at https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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
from towhee.models.poolformer.layernormchannel import LayerNormChannel
from towhee.models.metaformer.metaformerblock import MetaFormerBlock


def basic_blocks(dim, index, layers, token_mixer=nn.Identity,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=LayerNormChannel,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate MetaFormer blocks for a stage
    return: MetaFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(MetaFormerBlock(
            dim, token_mixer=token_mixer, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            ))
    blocks = nn.Sequential(*blocks)

    return blocks
