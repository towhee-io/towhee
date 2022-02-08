# Copyright 2021  Facebook. All rights reserved.
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
# This code is modified by Zilliz.


import torch
from torch import nn
from typing import List, Tuple

from towhee.models.layers.multi_scale_attention import MultiScaleAttention
from towhee.models.layers.droppath import DropPath
from towhee.models.layers.mlp import Mlp
from towhee.models.layers.pool_attention import AttentionPool


class MultiScaleBlock(nn.Module):
    """
    A multiscale vision transformer block.
    Each block contains a multiscale attention layer and a Mlp layer.
    ::
                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention        Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp                 Proj
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+
    Args:
        dim(int):
            Input feature dimension.
        dim_out(int):
            Output feature dimension.
        num_heads(int):
            Number of heads in the attention layer.
        mlp_ratio(float):
            MLP ratio which controls the feature dimension in the hidden layer of the MLP block.
        qkv_bias(bool):
            If set to False, the qkv layer will not learn an additive bias.
        dropout_rate(float):
            DropOut rate. If set to 0, DropOut is disabled.
        droppath_rate(float):
            DropPath rate. If set to 0, DropPath is disabled.
        activation(nn.Module):
            Activation layer used in the MLP layer.
        norm_layer(nn.Module):
            Normalization layer.
        kernel_q(_size_3_t):
            Pooling kernel size for q. If pooling kernel size is 1 for all the dimensions.
        kernel_kv(_size_3_t):
            Pooling kernel size for kv. If pooling kernel size is 1 for all the dimensions, pooling is not used.
        stride_q(_size_3_t):
            Pooling kernel stride for q.
        stride_kv(_size_3_t):
            Pooling kernel stride for kv.
        pool_mode(nn.Module):
            Pooling mode.
        has_cls_embed(bool):
            If set to True, the first token of the input tensor should be a cls token.
            Otherwise, the input tensor does not contain a cls token. Pooling is not applied to the cls token.
        pool_first(bool):
            If set to True, pool is applied before qkv projection. Otherwise, pool is applied after qkv projection.
    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        dropout_rate=0.0,
        droppath_rate=0.0,
        activation=nn.GELU,
        norm_layer=nn.LayerNorm,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        pool_mode=nn.Conv3d,
        has_cls_embed=True,
        pool_first=False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            pool_mode=pool_mode,
            pool_first=pool_first,
        )
        self.drop_path = (
            DropPath(drop_prob=droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim_out,
            act_layer=activation,
            drop=dropout_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(kernel_skip) > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x(torch.Tensor):
                Input tensor.
            thw_shape(List):
                The shape of the input tensor (before flattening).
        """

        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        atn = AttentionPool(
            pool=self.pool_skip,
            thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed
        )
        x_res, _ = atn(x)
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new
