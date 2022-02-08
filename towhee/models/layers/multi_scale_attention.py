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
import numpy

from towhee.models.layers.pool_attention import AttentionPool


class MultiScaleAttention(nn.Module):
    """
    A multiscale attention block.
    compare to a conventional attention block, a multiscale attention block optionally
    supports pooling (either before or after qkv projection). If pooling is not used, a
    multiscale attention block is equivalent to a conventional attention block.
    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                    → -------------- ←                 |
                             ↓                         |
                       MatMul & Scale                  |
                             ↓                         |
                          Softmax                      |
                             → ----------------------- ←
                                         ↓
                                   MatMul & Scale
                                         ↓
                                      DropOut
    Args:
        dim(int):
            Input feature dimension.
        num_heads(int):
            number of heads in the attention layer.
        qkv_bias(bool):
            If set to False, the qkv layer will not learn an additive bias.
        dropout_rate(float):
            Dropout rate.
        kernel_q(_size_3_t):
            Pooling kernel size for q. If both pooling kernel
            size and pooling stride size are 1 for all the dimensions, pooling is disabled.
        kernel_kv(_size_3_t):
            Pooling kernel size for kv. If both pooling kernel size and pooling stride size
            are 1 for all the dimensions, pooling is disabled.
        stride_q(_size_3_t):
            Pooling kernel stride for q.
        stride_kv(_size_3_t):
            Pooling kernel stride for kv.
        norm_layer(nn.Module):
            normalization layer used after pooling.
        has_cls_embed(bool):
            If set to True, the first token of the input tensor
            should be a cls token. Otherwise, the input tensor does not contain a cls token.
            Pooling is not applied to the cls token.
        pool_mode(str):
            Pooling mode. Option includes "conv" (learned pooling), "avg"
            (average pooling), and "max" (max pooling).
        pool_first(bool):
            If set to True, pool is applied before qkv projection.
            Otherwise, pool is applied after qkv projection.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        dropout_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        pool_mode=nn.Conv3d,
        pool_first=False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.kernel_q = kernel_q
        self.kernel_kv = kernel_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        self.norm_layer = norm_layer
        self.has_cls_embed = has_cls_embed
        self.pool_mode = pool_mode
        self.pool_first = pool_first

        assert self.pool_mode in [nn.Conv3d, nn.AvgPool3d, nn.MaxPool3d]

        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.padding_q = [int(q // 2) for q in self.kernel_q]
        self.padding_kv = [int(kv // 2) for kv in self.kernel_kv]

        self.q = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
        self.k = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
        self.v = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        if self.dropout_rate > 0.0:
            self.proj_drop = nn.Dropout(self.dropout_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
            self.kernel_q is not None
            and numpy.prod(self.kernel_q) == 1
            and numpy.prod(self.stride_q) == 1
        ):
            self.kernel_q = None
        if (
            self.kernel_kv is not None
            and numpy.prod(self.kernel_kv) == 1
            and numpy.prod(self.stride_kv) == 1
        ):
            self.kernel_kv = None

        if self.pool_mode in (nn.AvgPool3d, nn.MaxPool3d):
            pool_op = nn.MaxPool3d if pool_mode == nn.MaxPool3d else nn.AvgPool3d
            self.pool_q = (
                pool_op(self.kernel_q, self.stride_q, self.padding_q, ceil_mode=False)
                if self.kernel_q is not None
                else None
            )
            self.pool_k = (
                pool_op(self.kernel_kv, self.stride_kv, self.padding_kv, ceil_mode=False)
                if self.kernel_kv is not None
                else None
            )
            self.pool_v = (
                pool_op(self.kernel_kv, self.stride_kv, self.padding_kv, ceil_mode=False)
                if self.kernel_kv is not None
                else None
            )
        elif self.pool_mode == nn.Conv3d:
            self.pool_q = (
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    self.kernel_q,
                    stride=self.stride_q,
                    padding=self.padding_q,
                    groups=self.head_dim,
                    bias=False,
                )
                if self.kernel_q is not None
                else None
            )
            self.norm_q = self.norm_layer(self.head_dim) if self.kernel_q is not None else None
            self.pool_k = (
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    self.kernel_kv,
                    stride=self.stride_kv,
                    padding=self.padding_kv,
                    groups=self.head_dim,
                    bias=False,
                )
                if self.kernel_kv is not None
                else None
            )
            self.norm_k = self.norm_layer(self.head_dim) if self.kernel_kv is not None else None
            self.pool_v = (
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    self.kernel_kv,
                    stride=self.stride_kv,
                    padding=self.padding_kv,
                    groups=self.head_dim,
                    bias=False,
                )
                if self.kernel_kv is not None
                else None
            )
            self.norm_v = self.norm_layer(self.head_dim) if self.kernel_kv is not None else None
        else:
            raise NotImplementedError("Unsupported model.")

    def qkv_proj(
            self,
            q,
            q_size,
            k,
            k_size,
            v,
            v_size,
            batch_size,
            chan_size,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            q(torch.Tensor):
                q tensor.
            q_size(List[int]):
                q tensor size.
            k(torch.Tensor):
                k tensor.
            k_size(List[int]):
                k tensor size.
            v(torch.Tensor):
                v tensor.
            v_size(List[int]):
                v tensor size.
            batch_size(List[int]):
                batch size.
            chan_size(List[int]):
                channel size.
        """
        q = (
            self.q(q)
                .reshape(batch_size, q_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        k = (
            self.k(k)
                .reshape(batch_size, k_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        v = (
            self.v(v)
                .reshape(batch_size, v_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        return q, k, v

    def qkv_pool(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            thw_shape: Tuple[torch.Tensor, List[int]],
    ) -> Tuple[
        torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]
    ]:
        """
        Args:
            q(torch.Tensor):
                q tensor.
            k(torch.Tensor):
                k tensor.
            v(torch.Tensor):
                v tensor.
            thw_shape(Tuple[torch.Tensor, List[int]]):
                The shape of the input tensor.
        """
        ap = AttentionPool(
            pool=self.pool_q,
            thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        q, q_shape = ap(q)
        ap = AttentionPool(
            pool=self.pool_k,
            thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        k, k_shape = ap(k)
        ap = AttentionPool(
            pool=self.pool_v,
            thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )
        v, v_shape = ap(v)
        return q, q_shape, k, k_shape, v, v_shape

    def get_qkv_length(
            self,
            q_shape,
            k_shape,
            v_shape,
    ) -> Tuple[int]:
        """
        Args:
            q_shape(List[int]):
                q tensor shape.
            k_shape(List[int]):
                k tensor shape.
            v_shape(List[int]):
                v tensor shape.
        """
        q_n = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
        k_n = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
        v_n = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)
        return q_n, k_n, v_n

    def reshape_qkv_to_seq(
            self,
            q,
            k,
            v,
            q_n,
            v_n,
            k_n,
            b,
            c,
    ) -> Tuple[int]:
        """
        Args:
            q(torch.Tensor):
                q tensor.
            k(torch.Tensor):
                k tensor.
            v(torch.Tensor):
                v tensor.
            q_n(int):
                k tensor size.
            v_n(int):
                v tensor size.
            k_n(int):
                k tensor size.
            b(int):
                Reshaped size.
            c(int):
                Reshaped size.
        """
        q = q.permute(0, 2, 1, 3).reshape(b, q_n, c)
        v = v.permute(0, 2, 1, 3).reshape(b, v_n, c)
        k = k.permute(0, 2, 1, 3).reshape(b, k_n, c)
        return q, k, v

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

        b, n, c = x.shape
        if self.pool_first:
            x = x.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
            q = k = v = x
            q, q_shape, k, k_shape, v, v_shape = self.qkv_pool(q, k, v, thw_shape)
            q_n, k_n, v_n = self.get_qkv_length(q_shape, k_shape, v_shape)
            q, k, v = self.reshape_qkv_to_seq(q, k, v, q_n, v_n, k_n, b, c)
            q, k, v = self.qkv_proj(q, q_n, k, k_n, v, v_n, b, c)
        else:
            q = k = v = x
            q, k, v = self.qkv_proj(q, n, k, n, v, n, b, c)
            q, q_shape, k, k_shape, v, v_shape = self.qkv_pool(q, k, v, thw_shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        n = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        if self.dropout_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape
