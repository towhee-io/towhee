# Inspired by https://github.com/stdio2016/pfann
#
# Modifications by Copyright 2021 Zilliz. All rights reserved.
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

from typing import Union, Tuple


class SeparableConv2d(nn.Module):
    """
    Separable Conv2D in pytorch

    Args:
        in_c (`int`):
            Number of channels in the input image.
        out_c (`int`):
            Number of channels produced by the convolution.
        kernel_size (`Union[int, Tuple]`):
            Size of the convolving kernel.
        stride (`Union[int, Tuple]`):
            Stride of the convolution.
        in_f (`int`):
            Padding parameter.
        in_t (`int`):
            Padding parameter.
        fuller (`bool=False`):
            Whether to use group in conv2 layer.
        activation (`nn.Module=nn.ReLU()`):
            Activation layer.
        relu_after_bn (`bool=False`):
            Whether to add ReLU after batch norm.
    """
    def __init__(
            self,
            in_c: int,
            out_c: int,
            kernel_size: Union[int, Tuple],
            stride: Union[int, Tuple],
            in_f: int,
            in_t: int,
            fuller: bool = False,
            activation: nn.Module = nn.ReLU(),
            relu_after_bn: bool = False
    ):
        super().__init__()
        # Same padding
        padding = (in_t - 1) // stride[0] * stride[0] + kernel_size - in_t
        self.pad1 = nn.ZeroPad2d((padding // 2, padding - padding // 2, 0, 0))
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=(1, kernel_size), stride=(1, stride[0]))
        self.ln1 = nn.LayerNorm((out_c, in_f, (in_t - 1) // stride[0] + 1))
        self.relu1 = activation
        # Same padding
        padding = (in_f - 1) // stride[1] * stride[1] + kernel_size - in_f
        self.pad2 = nn.ZeroPad2d((0, 0, padding // 2, padding - padding // 2))
        if fuller:
            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(kernel_size, 1), stride=(stride[1], 1))
        else:
            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(kernel_size, 1), stride=(stride[1], 1), groups=out_c)
        self.ln2 = nn.LayerNorm((out_c, (in_f - 1) // stride[1] + 1, (in_t - 1) // stride[0] + 1))
        self.relu2 = activation

        self.relu_after_bn = relu_after_bn
        self.hacked = False

    # Equivalent to Keras Same Padding with stride=2
    def hack(self):
        self.hacked = not self.hacked
        with torch.no_grad():
            self.conv1.weight.set_(self.conv1.weight.flip([2, 3]))
            self.ln1.weight.set_(self.ln1.weight.flip([1, 2]))
            self.ln1.bias.set_(self.ln1.bias.flip([1, 2]))
            self.conv2.weight.set_(self.conv2.weight.flip([2, 3]))
            self.ln2.weight.set_(self.ln2.weight.flip([1, 2]))
            self.ln2.bias.set_(self.ln2.bias.flip([1, 2]))
            if self.hacked:
                self.conv1.padding = self.pad1.padding[3::-2]
                self.conv2.padding = self.pad2.padding[3::-2]
            else:
                self.conv1.padding = (0, 0)
                self.conv2.padding = (0, 0)

    def forward(self, x):
        if not self.hacked:
            x = self.pad1(x)
        x = self.conv1(x)
        if self.relu_after_bn:
            x = self.ln1(x)
            x = self.relu1(x)
        else:
            x = self.relu1(x)
            x = self.ln1(x)
        if not self.hacked:
            x = self.pad2(x)
        x = self.conv2(x)
        if self.relu_after_bn:
            x = self.ln2(x)
            x = self.relu2(x)
        else:
            x = self.relu2(x)
            x = self.ln2(x)
        return x
