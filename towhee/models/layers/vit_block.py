# Original pytorch implementation by:
# 'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
#       - https://arxiv.org/abs/2010.11929
# 'How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers'
#       - https://arxiv.org/abs/2106.10270
#
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

from torch import nn
from towhee.models.layers.multi_head_attention import MultiHeadAttention
from towhee.models.layers.droppath import DropPath
from towhee.models.layers.mlp import Mlp

class Block(nn.Module):
    """
    The Transformer block.
    Args:
        dim (int): number of features
        num_heads (int): number of heads
        mlp_ratio (int): ratio of mlp layer
        qkv_bias (bool): if add bias to qkv layer
        qk_scale (float): number to scale qk
        drop_ratio (float): drop rate at the end of the block (mlp layer)
        attn_drop_ratio (float): drop rate of attention layer
        drop_path_ratio (float): drop rate of drop_path layer
        act_layer: activation layer
        norm_layer: normalization layer
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
        self.drop_path = DropPath(drop_prob=drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# if __name__=='__main__':
#     import torch
#     x = torch.rand(1, 197, 768)
#     model = Block(dim=768, num_heads=8)
#     out = model.forward(x)
#     print(out.shape)
