# Implementation of utils for Shunted Transform in the paper:
#   [Shunted Self-Attention via Multi-Scale Token Aggregation](https://arxiv.org/abs/2111.15193)
#
# Inspired by original code from https://github.com/OliverRensu/Shunted-Transformer
#
# Additions & modifications are protected by Copyright 2021 Zilliz. All rights reserved.
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
from torch import nn

from towhee.models.layers.patch_embed2d import PatchEmbed2D
from towhee.models.layers.mlp import Mlp as ViTMlp
from towhee.models.layers.droppath import DropPath
from towhee.models.utils.init_vit_weights import init_vit_weights


class OverlapPatchEmbed(PatchEmbed2D):
    """
    Overlap patch embedding layer based on a standard patch embedding layer in ViT

    Args:
        img_size (`int`): image size (width or height)
        patch_size (`int`): patch size
        stride (`int`): stride used in convolution
        in_chans (`int`): number of input channels
        embed_dim (`int`): output dimension
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=nn.LayerNorm, flatten=True)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.apply(init_vit_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, h, w


class HeadPatchEmbed(nn.Module):
    """
    The patch embedding layer at stage 0.

    Args:
        num (`int`): number of repeated sets of layers
    """
    def __init__(self, num, patch_size=7, stride=2, in_chans=3, embed_dim=64):
        super().__init__()
        stem = [nn.Conv2d(in_chans, embed_dim, patch_size, stride, padding=3, bias=False),
                nn.BatchNorm2d(embed_dim), nn.ReLU(True)]
        for _ in range(num):
            stem.append(nn.Conv2d(embed_dim, embed_dim, 3, 1, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(embed_dim))
            stem.append(nn.ReLU(True))
        stem.append(nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=stride))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(init_vit_weights)

    def forward(self, x):
        x = self.conv(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class Mlp(ViTMlp):
    """
    Modified Mlp containing depthwise convolution, based on a standard Mlp in ViT.

    Args:
        in_features (`int`): input dimension
        hidden_features (`int`): dimension used in depthwise conv layer
        out_features (`int`): output dimension
        act_layer (`nn.Module`): activation layer
        drop (`float`): dropout rate
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            drop=drop
        )
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.apply(init_vit_weights)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, _, c = x.shape

        x0 = x.transpose(1, 2).view(b, c, h, w)
        x0 = self.dwconv(x0)
        x0 = x0.flatten(2).transpose(1, 2)

        x = self.act(x + x0)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Modified Attention.

    Args:
        dim (`int`): feature dimension
        num_heads (`int`): number of attention heads
        qkv_bias (`bool`): flag to use bias for qkv
        qk_scale (`int`): Number to scale qk
        attn_drop (`float`): drop rate of attention
        proj_drop (`float`): drop rate of projection
        sr_ratio (`int`): shunt rate
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divisible by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio in [8, 4, 2]:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio // 2, stride=sr_ratio // 2)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
            self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(init_vit_weights)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(b, c, -1).permute(0, 2, 1)))
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(b, c, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1).reshape(b, -1, 2, self.num_heads // 2, c // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x_2).reshape(b, -1, 2, self.num_heads // 2, c // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]
            k2, v2 = kv2[0], kv2[1]
            attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(b, -1, c // 2).
                                       transpose(1, 2).view(b, c // 2, h // self.sr_ratio, w // self.sr_ratio)). \
                view(b, c // 2, -1).view(b, self.num_heads // 2, c // self.num_heads, -1).transpose(-1, -2)
            x1 = (attn1 @ v1).transpose(1, 2).reshape(b, n, c // 2)
            attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(b, -1, c // 2).transpose(1, 2).
                                       view(b, c // 2, h * 2 // self.sr_ratio, w * 2 // self.sr_ratio)). \
                view(b, c // 2, -1).view(b, self.num_heads // 2, c // self.num_heads, -1).transpose(-1, -2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(b, n, c // 2)

            x = torch.cat([x1, x2], dim=-1)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(b, n, c) + \
                self.local_conv(v.transpose(1, 2).reshape(b, n, c).
                                transpose(1, 2).view(b, c, h, w)).view(b, c, n).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Block in Shunted Transformer

    Args:
        dim (`int`): feature dimension
        num_heads (`int`): number of attention heads
        mlp_ratio (`float`): number to scale hidden dimension in mlp
        qkv_bias (`bool`): flag to use bias in qkv
        qk_scale (`int`): number to scale qk
        drop (`float`): drop rate of Mlp
        attn_drop (`float`): drop rate of attention
        drop_path (`float`): drop rate of drop path
        act_layer (`nn.Module`): activation layer
        norm_layer (`nn.Module`): normalization layer
        sr_ratio (`int`): shunt rate
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(init_vit_weights)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))

        return x
