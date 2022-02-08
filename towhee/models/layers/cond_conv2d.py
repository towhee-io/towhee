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
import math
from typing import Callable, Tuple
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from towhee.models.utils.general_utils import to_2tuple
from towhee.models.layers.conv2d_same import conv2d_same
from towhee.models.layers.padding_functions import get_padding_value

def get_condconv_initializer(initializer: Callable, num_experts: int, expert_shape: Tuple[int, int, int, int]):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or
                weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))
    return condconv_initializer

class CondConv2d(nn.Module):
    """
    Conditionally Parameterized Convolution

    Inspired by: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py
    Grouped convolution hackery for parallel execution of the per-sample kernel filters inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983

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
        groups (`int`):
            Number of blocked connections from input channels to output channels.
        bias (`bool`):
            If True, adds a learnable bias to the output.
        num_experts (`int`):
            The number of expert kernels and biases.
    """
    __constants__ = ['in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3,
                 stride: int=1, padding: str='', dilation: int=1, groups: int=1, bias: bool=False, num_experts: int=4) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic  # if in forward to work with torchscript
        self.padding = to_2tuple(padding_val)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (b * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(b * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        x = x.view(1, b * c, h, w)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        out = out.permute([1, 0, 2, 3]).view(b, self.out_channels, out.shape[-2], out.shape[-1])
        return out

