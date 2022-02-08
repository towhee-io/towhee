# Copyright 2021  Facebook. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICEnSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" bASIS,
# wIthOUt wARRAntIES OR COnDItIOnS OF AnY KInD, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This code is modified by Zilliz.


import torch
from torch import nn
from typing import List, Tuple


class AttentionPool(nn.Module):
    """
    A MLP block that contains two linear layers with a normalization layer.
    the MLP block is used in a transformer model after the attention block.
    ::
                                        Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          norm
    Args:
        thw_shape(List):
            the shape of the input tensor (before flattening).
        pool(Callable):
            Pool operation that is applied to the input tensor.
            If pool is None, return the input tensor.
        has_cls_embed(bool):
            whether the input tensor contains cls token. Pool operation excludes cls token.
        norm(Callable):
            Optional normalization operation applied to tensor after pool.
    Returns:
        tensor(torch.Tensor):
            Input tensor after pool.
        thw_shape(List[int]):
            Output tensor shape (before flattening).
    """
    def __init__(
        self,
        thw_shape,
        pool=None,
        has_cls_embed=True,
        norm=None
    ) -> None:
        super().__init__()
        self.pool = pool
        self.thw_shape = thw_shape
        self.has_cls_embed = has_cls_embed
        self.norm = norm

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        if self.pool is None:
            return x, self.thw_shape
        tensor_dim = x.ndim
        if tensor_dim == 4:
            pass
        elif tensor_dim == 3:
            x = x.unsqueeze(1)
        else:
            raise NotImplementedError(f"Unsupported input dimension {x.shape}")

        if self.has_cls_embed:
            cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]

        b, n, _, c = x.shape
        t, h, w = self.thw_shape
        x = x.reshape(b * n, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

        x = self.pool(x)

        thw_shape = [x.shape[2], x.shape[3], x.shape[4]]
        l_pooled = x.shape[2] * x.shape[3] * x.shape[4]
        x = x.reshape(b, n, c, l_pooled).transpose(2, 3)
        if self.has_cls_embed:
            x = torch.cat((cls_tok, x), dim=2)
        if self.norm is not None:
            x = self.norm(x)

        if tensor_dim == 4:
            pass
        else:  # For the case tensor_dim == 3.
            x = x.squeeze(1)
        return x, thw_shape
