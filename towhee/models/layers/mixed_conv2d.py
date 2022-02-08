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
from typing import List

import torch
from torch import nn

from towhee.models.utils.create_conv2d_pad import create_conv2d_pad

def _split_channels(num_chan: int, num_groups: int) -> List[int]:
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

class MixedConv2d(nn.ModuleDict):
    """
    Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py

    Args:
        in_channels (`int`):
            Number of channels in the input image
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`):
            Size of the convolving kernel.
        stride (`int`):
            Stride of the convolution.
        padding (`str`):
            Padding added to all four sides of the input.
        dilation (`int`):
            Spacing between kernel elements.
        depthwise (`bool`):
            If True, use depthwise convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3,
                 stride: int=1, padding: str='', dilation: int=1, depthwise: bool=False, **kwargs) -> None:
        super().__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x
