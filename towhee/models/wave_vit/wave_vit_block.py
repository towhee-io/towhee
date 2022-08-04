# Original pytorch implementation by:
# 'Wave-ViT: Unifying Wavelet and Transformers for Visual Representation Learning'
#       - http://arxiv.org/pdf/2207.04978
# Original code by / Copyright 2022, YehLi.
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
import torch
from torch import nn
from timm.models.layers import DropPath
import numpy as np
from towhee.models.wave_vit.wave_vit_utils import DWT2D, IDWT2D
from towhee.models.utils.init_vit_weights import init_vit_weights
from towhee.models.layers.mlp import Mlp


def rand_bbox(size, lam, scale=1):
    w = size[1] // scale
    h = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(w * cut_rat)
    cut_h = np.int64(h * cut_rat)

    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


class DWConv(nn.Module):
    """
    DWConv
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dw_conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ClassAttention(nn.Module):
    """
    ClassAttention
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2)
        self.q = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.apply(init_vit_weights)

    def forward(self, x):
        b, n, c = x.shape
        kv = self.kv(x).reshape(b, n, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x[:, :1, :]).reshape(b, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        cls_embed = (attn @ v).transpose(1, 2).reshape(b, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):
    """
    ClassBlock
    """
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = ClassAttention(dim, num_heads)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio))
        self.apply(init_vit_weights)

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.attn(self.norm1(x))
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


class PVT2FFN(nn.Module):
    """
    PVT2FFN
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dw_conv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(init_vit_weights)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dw_conv(x, h, w)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WaveAttention(nn.Module):
    """
    WaveAttention
    """
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWT2D(wave="haar")
        self.i_dwt = IDWT2D(wave="haar")
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.proj = nn.Linear(dim + dim // 4, dim)
        self.apply(init_vit_weights)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)

        x_idwt = self.i_dwt(x_dwt)
        x_idwt = x_idwt.view(b, -1, x_idwt.size(-2) * x_idwt.size(-1)).transpose(1, 2)

        kv = self.kv_embed(x_dwt).reshape(b, c, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        return x


class Attention(nn.Module):
    """
    Attention
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(init_vit_weights)

    def forward(self, x, h, w):
        _, _ = h, w
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return x


class Block(nn.Module):
    """
    Block
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 block_type="wave"
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if block_type == "std_att":
            self.attn = Attention(dim, num_heads)
        else:
            self.attn = WaveAttention(dim, num_heads, sr_ratio)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(init_vit_weights)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class DownSamples(nn.Module):
    """
    DownSamples
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(init_vit_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class Stem(nn.Module):
    """
    Stem
    """
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(init_vit_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w

