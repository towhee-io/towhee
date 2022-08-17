# Implementation of "Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrasive Learning."
# https://arxiv.org/abs/2010.11910
#
# Inspired by https://github.com/stdio2016/pfann
#
# Additions & Modifications by Copyright 2021 Zilliz. All rights reserved.
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

from towhee.models.layers.conv2d_separable import SeparableConv2d


class FrontConv(nn.Module):
    """
    Front Convolutional Layers

    Args:
        dim (`int`):
            Dimension of features
        h (`int`):
            Height of input.
        in_f (`int`):
            Padding parameter.
        in_t (`int`):
            Padding parameter.
        fuller (`bool=False`):
            Whether to use group in conv2 layer.
        activation (`str`):
            Activation layer.
        strides (`list of tuple`):
            A list of stride tuples.
        relu_after_bn (`bool`):
            Whether to add ReLU after batch norm.
    """
    def __init__(self, dim, h, in_f, in_t,
                 fuller=False, activation='relu',
                 strides=None, relu_after_bn=True):
        super().__init__()
        channels = [1, dim, dim, 2 * dim, 2 * dim, 4 * dim, 4 * dim, h, h]
        convs = []
        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'elu':
            activation = nn.ELU()
        else:
            raise ValueError(f'Invalid activation "{activation}". Accept "relu" or "elu" only.')
        for i in range(8):
            kernel_size = 3
            stride = (2, 2)
            if strides is not None:
                stride = strides[i][0][1], strides[i][1][0]
            sep_conv = SeparableConv2d(
                in_c=channels[i],
                out_c=channels[i + 1],
                kernel_size=kernel_size,
                stride=stride,
                in_f=in_f, in_t=in_t,
                fuller=fuller, activation=activation, relu_after_bn=relu_after_bn
            )
            convs.append(sep_conv)
            in_f = (in_f - 1) // stride[1] + 1
            in_t = (in_t - 1) // stride[0] + 1
        assert in_f == in_t == 1, 'output must be 1x1'
        self.convs = nn.ModuleList(convs)

    def hack(self):
        for conv in self.convs:
            conv.hack()

    def forward(self, x):
        x = x.unsqueeze(1)
        for _, conv in enumerate(self.convs):
            x = conv(x)
        return x


class DivEncoder(nn.Module):
    """
    Divider & encoder

    dim (`int`):
        Dimension of features
    h (`int`):
        Height of input.
    u (`int`):
        Parameter to multiple linear dimension.
    """
    def __init__(self, dim, h, u):
        super().__init__()
        assert h % dim == 0, f'h ({h}) must be divisible by d ({dim})'
        v = h // dim
        self.d = dim
        self.h = h
        self.u = u
        self.v = v
        self.linear1 = nn.Conv1d(dim * v, dim * u, kernel_size=(1,), groups=dim)
        self.elu = nn.ELU()
        self.linear2 = nn.Conv1d(dim * u, dim, kernel_size=(1,), groups=dim)

    def forward(self, x, norm=True):
        x = x.reshape([-1, self.h, 1])
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        x = x.reshape([-1, self.d])
        if norm:
            x = torch.nn.functional.normalize(x, p=2.0)
        return x


class NNFp(nn.Module):
    """
    Neural network Fingerprinter

    Args:
        dim (`int`):
            Dimension of features
        h (`int`):
            Height of input.
        u (`int`):
            Parameter to multiple linear dimension.
        in_f (`int`):
            Padding parameter.
        in_t (`int`):
            Padding parameter.
        fuller (`bool=False`):
            Whether to use group in conv2 layer.
        activation (`str`):
            Activation layer.
        strides (`list of tuple`):
            A list of stride tuples.
        relu_after_bn (`bool`):
            Whether to add ReLU after batch norm.
    """
    def __init__(self, dim, h, u, in_f, in_t, fuller=False, activation='relu', strides=None, relu_after_bn=True):
        super().__init__()
        self.front = FrontConv(
            dim, h, in_f, in_t,
            fuller=fuller,
            activation=activation,
            strides=strides,
            relu_after_bn=relu_after_bn
            )
        self.div_encoder = DivEncoder(dim, h, u)
        self.hacked = False

    def hack(self):
        self.hacked = not self.hacked
        self.front.hack()

    def forward(self, x, norm=True):
        if self.hacked:
            x = x.flip([1, 2])
        x = self.front(x)
        x = self.div_encoder(x, norm=norm)
        return x
