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

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.
    Args:
        dim (int): number of features
        num_heads (int): number of heads
        qkv_bias (bool): if add bias to qkv layer
        qk_scale (float): number to scale qk
        attn_drop_ratio (float): drop rate of attention layer
        proj_drop_ratio (float): drop rate of projection layer
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0,
                 proj_drop_ratio=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        batch_size, new_num_patch, dim = x.shape

        qkv = self.qkv(x).reshape(
            batch_size,
            new_num_patch,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, new_num_patch, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# if __name__=='__main__':
#     import torch
#
#     input_x = torch.rand(1, 196+1, 768) # shape of output from patch_embed
#     model = MultiHeadAttention(dim=768)
#     out = model.forward(input_x)
#
#     print(out.shape)
