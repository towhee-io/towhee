# Built on top of the original implementation at https://github.com/sail-sg/poolformer/blob/main/models/metaformer.py
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
import torch

from towhee.models.layers.droppath import DropPath
from towhee.models.poolformer.layernormchannel import LayerNormChannel
from towhee.models.poolformer.mlp import Mlp


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    Args:
        dim (int): embedding dimension
        mlp_ratio (float): mlp ratio
        act_layer (nn.module): activation layer
        norm_layer (nn.module): normalization layer
        drop (float): drop out rate
        drop_path (float): stochastic depth
        use_layer_scale (bool): use layer scale
    """
    def __init__(self,
                 dim,
                 token_mixer=nn.Identity,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=LayerNormChannel,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
