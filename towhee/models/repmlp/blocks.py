# Pytorch implementation of [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality]
# (https://arxiv.org/abs/2112.11081)
#
# Inspired by https://github.com/DingXiaoH/RepMLP
#
# Additions & modifications by Copyright 2021 Zilliz. All rights reserved.
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


class GlobalPerceptron(nn.Module):
    """
    Global Perception Block

    Args:
        - input_channels (`int`):
            Number of input channels & final output channels.
        - internal_neurons (`int`):
            Number of channels used to connect conv2d layers inside block.

    Example:
        >>> import torch
        >>> from towhee.models.repmlp import GlobalPerceptron
        >>>
        >>> data = torch.rand(3, 1, 1)
        >>> layer = GlobalPerceptron(input_channels=3, internal_neurons=4)
        >>> out = layer(data)
        >>> print(out.shape)
        torch.Size([1, 3, 1, 1])
    """
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = nn.functional.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.fc1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return x
