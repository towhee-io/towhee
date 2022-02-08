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


class ResNetBasic3DModule(nn.Module):
    """
    ResNet basic 3D stem module. It performs spatiotemporal Convolution, BN, and activation
    following by a spatiotemporal pooling.
                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d
    Args:
        conv (`nn.Module`):
            Convolutional module.
        norm (`nn.Module`):
            Normalization module.
        activation (`nn.Module`):
            Activation module.
        pool (`nn.Module`):
            Pooling module.
    """

    def __init__(
        self,
        *,
        conv=None,
        norm=None,
        activation=None,
        pool=None,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.activation = activation
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x
