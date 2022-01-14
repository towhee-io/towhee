# Copyright 2021 Ross Wightman and Zilliz. All rights reserved.
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
from towhee.models.layers.multi_head_attention import MultiHeadAttention
from towhee.models.layers.droppath import DropPath
from towhee.models.layers.mlp import Mlp

class Block(nn.Module):
    """
    Block in VisionTransformer includes multi_head_attention and mlp
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0,
                 attn_drop_ratio=0,
                 drop_path_ratio=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio
        )
        self.drop_path = DropPath(drop_prob=drop_path_ratio) if drop_path_ratio >0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        a = x + self.drop_path(self.attn(self.norm1(x)))
        x = a + self.drop_path(self.mlp(self.norm2(x)))
        return x
