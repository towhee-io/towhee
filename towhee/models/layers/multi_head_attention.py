# Original pytorch implementation by:
# 'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
#       - https://arxiv.org/abs/2010.11929
# 'How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers'
#       - https://arxiv.org/abs/2106.10270
#
# Built on top of codes from / Copyright 2020, Ross Wightman & Facebook, Inc. and its affiliates.
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

import os

try:
    # pylint: disable=unused-import
    import einops
except ImportError:
    os.system("pip install einops")

from torch import nn

from einops import rearrange
from towhee.models.layers.layers_with_relprop import Einsum, Linear, Dropout, Softmax


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.

    Args:
        dim (`int`):
            number of features
        num_heads (`int=8`):
            number of heads
        qkv_bias (`bool=False`):
            if add bias to qkv layer
        qk_scale (`float=None`):
            number to scale qk
        attn_drop_ratio (`float=0.`):
            drop rate of attention layer
        proj_drop_ratio (`float=0.`):
            drop rate of projection layer
        with_qkv (`bool=True`):
            if use qkv layer

    Example:
        >>> import torch
        >>> from towhee.models.layers.multi_head_attention import MultiHeadAttention
        >>>
        >>> test_shape = (1, 196+1, 768)  # shape of output from patch_embed
        >>> input_x = torch.rand(test_shape)
        >>> model = MultiHeadAttention(dim=test_shape[2])
        >>> out = model.forward(input_x)
        >>> print(out.shape)
        torch.Size([1, 197, 768])
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = Einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = Einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop_ratio)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop_ratio)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

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

        self.save_v(v)

        self.matmul1([q, k])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        self.matmul2([attn, v])
        x = (attn @ v).transpose(1, 2).reshape(batch_size, new_num_patch, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


# if __name__ == '__main__':
#     import torch
#
#     test_shape = (1, 196+1, 768)
#     input_x = torch.rand(test_shape)  # shape of output from patch_embed
#     model = MultiHeadAttention(dim=test_shape[2])
#     out = model.forward(input_x)
#
#     assert(out.shape == (1, 197, 768))
