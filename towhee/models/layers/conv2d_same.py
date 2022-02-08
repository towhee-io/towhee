# Copyright 2021 Ross Wightman . All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This code is modified by Zilliz.
from typing import Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

from towhee.models.layers.padding_functions import pad_same

def conv2d_same(
       x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    """
    Tensorflow like 'SAME' convolution function for 2D convolutions.
    """
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    _ = padding
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    """
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions.

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`Union[int, Tuple]`):
            Size of the convolving kernel.
        stride (`Union[int, Tuple]`):
            Stride of the convolution.
        padding (`Union[int, Tuple, str]`):
            Padding added to all four sides of the input.
        dilation (`int`):
            Spacing between kernel elements.
        groups (`int`):
            Number of blocked connections from input channels to output channels.
        bias (`bool`):
            If True, adds a learnable bias to the output.
    """

    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        _ = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

