# Built on top of the original implementation at https://github.com/youngwanLEE/MPViT
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


import math
import os
import sys
from functools import partial
from typing import List, Dict

import numpy as np
import torch

try:
    # pylint: disable=unused-import
    import einops
except ImportError:
    os.system("pip install einops")

from einops import rearrange
from towhee.models.layers.droppath import DropPath
from towhee.models.utils.weight_init import trunc_normal_
from torch import einsum, nn

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

url_dict = {
    "mpvit_tiny": "https://dl.dropbox.com/s/1cmquqyjmaeeg1n/mpvit_tiny.pth",
    "mpvit_xsmall": "https://dl.dropbox.com/s/vvpq2m474g8tvyq/mpvit_xsmall.pth",
    "mpvit_small": "https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth",
    "mpvit_base": "https://dl.dropbox.com/s/la8w31m0apj2830/mpvit_base.pth",
}


def _cfg_mpvit(url="", **kwargs):
    """configuration of mpvit."""
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a.

    MLP) class.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """forward function"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2dBN(nn.Module):
    """Convolution with BN module."""

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        """forward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)
        return x


class DWConv2dBN(nn.Module):
    """Depthwise Separable Convolution with BN module."""

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        """
        forward function
        """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2dBN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )

    def forward(self, x):
        """forward function"""
        x = self.patch_conv(x)

        return x


class PatchEmbedStage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, embed_dim, num_path=4, is_pool=False):
        super().__init__()
        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=2 if is_pool and idx == 0 else 1,
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        """forward function"""
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.

    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        """init function"""

        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        """forward function"""
        bb, _, cc = x.shape
        hh, ww = size

        feat = x.transpose(1, 2).view(bb, cc, hh, ww)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""

    def __init__(self, ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * ch,
                cur_head_split * ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * ch for x in self.head_splits]

    def forward(self, q, v, size):
        """forward function"""
        _, h, _, _ = q.shape
        hh, ww = size

        # We don"t use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [bb, h, hh*ww, ch] -> [bb, h*ch, hh, ww].
        v_img = rearrange(v_img, "bb h (hh ww) ch -> bb (h ch) hh ww", hh=hh, ww=ww)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [bb, h*ch, hh, ww] -> [bb, h, hh*ww, ch].
        conv_v_img = rearrange(conv_v_img, "bb (h ch) hh ww -> bb h (hh ww) ch", h=h)

        ev_hat_img = q_img * conv_v_img
        ev_hat = ev_hat_img
        return ev_hat


class FactorAttConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        """forward function"""
        bb, n, cc = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(bb, n, 3, self.num_heads,
                                   cc // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        k_softmax_t_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_t_dot_v)

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(bb, n, cc)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MHCABlock(nn.Module):
    """Multi-Head Convolutional self-Attention block."""

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            drop_path=0.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAttConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        """forward function"""
        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x


class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""

    # pylint: disable=dangerous-default-value
    def __init__(
            self,
            dim: int,
            num_layers: int = 1,
            num_heads: int = 8,
            mlp_ratio: int = 3,
            drop_path_list: List = [],
            qk_scale: float = None,
            crpe_window: Dict = {
                3: 2,
                5: 3,
                7: 3
            },
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(ch=dim // num_heads,
                                  h=num_heads,
                                  window=crpe_window)
        self.MHCA_layers = nn.ModuleList([  # pylint: disable=invalid-name
            MHCABlock(
                dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_list[idx],
                qk_scale=qk_scale,
                shared_cpe=self.cpe,
                shared_crpe=self.crpe,
            ) for idx in range(self.num_layers)
        ])

    def forward(self, x, size):
        """forward function"""
        hh, ww = size
        bb = x.shape[0]
        for layer in self.MHCA_layers:
            x = layer(x, (hh, ww))

        # return x"s shape : [bb, N, C] -> [bb, C, hh, ww]
        x = x.reshape(bb, hh, ww, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2dBN(in_features,
                              hidden_features,
                              act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2dBN(hidden_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        """forward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MHCAStage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
    layers."""

    # pylint: disable=dangerous-default-value
    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            num_path=4,
            drop_path_list=[],
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList([
            MHCAEncoder(
                embed_dim,
                num_layers,
                num_heads,
                mlp_ratio,
                drop_path_list=drop_path_list,
            ) for _ in range(num_path)
        ])

        self.InvRes = ResBlock(in_features=embed_dim, out_features=embed_dim)  # pylint: disable=invalid-name
        self.aggregate = Conv2dBN(embed_dim * (num_path + 1),
                                  out_embed_dim,
                                  act_layer=nn.Hardswish)

    def forward(self, inputs):
        """forward function"""
        att_outputs = [self.InvRes(inputs[0])]
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, hh, ww] -> [B, N, C]
            _, _, hh, ww = x.shape
            x = x.flatten(2).transpose(1, 2)
            att_outputs.append(encoder(x, size=(hh, ww)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)

        return out


class ClsHead(nn.Module):
    """a linear layer for classification."""

    def __init__(self, embed_dim, num_classes):
        """initialization"""
        super().__init__()

        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """forward function"""
        # (B, C, H, W) -> (B, C, 1)

        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        # Shape : [B, C]
        out = self.cls(x)
        return out


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """Generate drop path rate list following linear decay rule."""
    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


class MPViT(nn.Module):
    """
    Multi-Path ViT class.

    Args:
        img_size (`int`):
            Input images size.
        num_stages (`int`):
            Network stage numbers.
        num_path (`List`):
            Path number in every stage.
        num_layers (`List`):
            Layers in every stage.
        embed_dims (`List`):
            Embed dim in every stage.
        mlp_ratios (`List`):
            MLP ratio in every stage.
        num_heads (`List`):
            Head number in every stage.
        drop_path_rate (`float`):
            Drop path rate.
        in_chans (`int`):
            Input channels.
        num_classes (`int`):
            Output classes number.
    """
    # pylint: disable=dangerous-default-value
    def __init__(
            self,
            img_size: int = 224,
            num_stages: int = 4,
            num_path: List = [4, 4, 4, 4],
            num_layers: List = [1, 1, 1, 1],
            embed_dims: List = [64, 128, 256, 512],
            mlp_ratios: List = [8, 8, 4, 4],
            num_heads: List = [8, 8, 8, 8],
            drop_path_rate: float = 0.0,
            in_chans: int = 3,
            num_classes: int = 1000,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_stages = num_stages

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2dBN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2dBN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList([
            PatchEmbedStage(
                embed_dims[idx],
                num_path=num_path[idx],
                is_pool=False if idx == 0 else True,  # pylint: disable=simplifiable-if-expression
            ) for idx in range(self.num_stages)
        ])

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList([
            MHCAStage(
                embed_dims[idx],
                embed_dims[idx + 1]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                num_layers[idx],
                num_heads[idx],
                mlp_ratios[idx],
                num_path[idx],
                drop_path_list=dpr[idx],
            ) for idx in range(self.num_stages)
        ])

        # Classification head.
        self.cls_head = ClsHead(embed_dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        """get classifier function"""
        return self.head

    def forward_features(self, x):
        """forward feature function"""

        # x"s shape : [B, C, H, W]

        x = self.stem(x)  # Shape : [B, C, H/4, W/4]

        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)
            x = self.mhca_stages[idx](att_inputs)

        return x

    def forward(self, x):
        """forward function"""
        x = self.forward_features(x)

        # cls head
        out = self.cls_head(x)
        return out


def mpvit_tiny(**kwargs):
    """mpvit_tiny :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 96, 176, 216]
    - MLP_ratio : 2
    Number of params: 5843736
    FLOPs : 1654163812
    Activations : 16641952
    """

    model = MPViT(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 96, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_xsmall(**kwargs):
    """mpvit_xsmall :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 128, 192, 256]
    - MLP_ratio : 4
    Number of params : 10573448
    FLOPs : 2971396560
    Activations : 21983464
    """

    model = MPViT(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 128, 192, 256],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_small(**kwargs):
    """mpvit_small :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 6, 3]
    - #channels : [64, 128, 216, 288]
    - MLP_ratio : 4
    Number of params : 22892400
    FLOPs : 4799650824
    Activations : 30601880
    """

    model = MPViT(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_base(**kwargs):
    """mpvit_base :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 8, 3]
    - #channels : [128, 224, 368, 480]
    MLP_ratio : 4
    Number of params: 74845976
    FLOPs : 16445326240
    Activations : 60204392
    """

    model = MPViT(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 8, 3],
        embed_dims=[128, 224, 368, 480],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def create_model(
        model_name: str = None,
        num_classes: int = 1000,
        pretrained: bool = False,
        weights_path: str = None,
        device: str = None,
) -> MPViT:
    """
    Create MViT model.
    Args:
        model_name (`str`):
            Name of MPViT model, it can be `mpvit_tiny`, `mpvit_xsmall`, `mpvit_small` or `mpvit_base`.
        num_classes (`int`):
            Classification head in the model, default is 1000, for the default pretrained model is pretrained in ImageNet1k.
        pretrained (`bool`):
            Whether the model using pretrained weights, default is None.
        weights_path (`str`):
            Local weights path.
        device (`str`):
            Model device, `cpu` or `cuda`

    Returns:
        (`MPViT`)
            MPViT model.

    >>> from towhee.models import mpvit
    >>> model = mpvit.create_model('mpvit_tiny')
    >>> model.__class__.__name__
    'MPViT'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name is None:
        raise AssertionError("no model name is specified.")
    else:
        current_module = sys.modules[__name__]
        model_func = getattr(current_module, model_name)
        model = model_func(num_classes=num_classes)
        if pretrained:
            if weights_path:
                checkpoint = torch.load(weights_path, map_location="cpu")
            else:
                url = url_dict[model_name]
                checkpoint = torch.hub.load_state_dict_from_url(
                    url, map_location="cpu", check_hash=True
                )
            model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
    return model
