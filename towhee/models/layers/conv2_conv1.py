# Copyright 2021 Zilliz and Facebook. All rights reserved.
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


class Conv2and1D(nn.Module):
    """
    Implementation of 2+1d Convolution by factorizing 3D Convolution into an 1D temporal
    Convolution and a 2D spatial Convolution with Normalization and Activation module
    in between:
    ::
                        Conv_t (or Conv_xy if conv_xy_first = True)
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                        Conv_xy (or Conv_t if conv_xy_first = True)
    The 2+1d Convolution is used to build the R(2+1)D network.
    """

    def __init__(
        self,
        *,
        conv_t=None,
        norm=None,
        activation=None,
        conv_xy=None,
        conv_xy_first=False,
    ) -> None:
        """
        Args:
            conv_t (`nn.Module`):
                Temporal convolution module.
            norm (`nn.Module`):
                Normalization module.
            activation (`nn.Module`):
                Activation module.
            conv_xy (`nn.Module`):
                Spatial convolution module.
            conv_xy_first (`bool`):
                If True, spatial convolution comes before temporal conv.
        """
        super().__init__()
        self.conv_t = conv_t
        self.norm = norm
        self.activation = activation
        self.conv_xy = conv_xy
        self.conv_xy_first = conv_xy_first
        assert self.conv_t is not None
        assert self.conv_xy is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_xy(x) if self.conv_xy_first else self.conv_t(x)
        x = self.norm(x) if self.norm else x
        x = self.activation(x) if self.activation else x
        x = self.conv_t(x) if self.conv_xy_first else self.conv_xy(x)
        return x
