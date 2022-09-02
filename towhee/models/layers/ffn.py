# Inspired by https://github.com/DingXiaoH/RepMLP
#
# Copyright 2022 Zilliz. All rights reserved.
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

from torch import nn

from towhee.models.layers.conv_bn_activation import Conv2dBNActivation


class FFNBlock(nn.Module):
    """
    The common FFN block.

    Args:
        in_channels (`int`):
            Number of input channels.
        hidden_channels (`int`):
            Number of hidden channels to connect conv2d layers.
        out_channels (`int`):
            Number of output channels.
        act_layer (`nn.Module`):
            Activation layer.

    Example:
        >>> import torch
        >>> from towhee.models.layers.ffn import FFNBlock
        >>>
        >>> x = torch.randn(1, 3, 1, 3)
        >>> layer = FFNBlock(in_channels=3, hidden_channels=2, out_channels=2)
        >>> y = layer(x)
        >>> print(y.shape)
        torch.Size([1, 2, 1, 3])
    """
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = Conv2dBNActivation(
            in_channels, hidden_features, kernel_size=1, padding=0, stride=1,
            norm_layer=nn.BatchNorm2d, eps=1e-5
        )
        self.ffn_fc2 = Conv2dBNActivation(
            hidden_features, out_features, kernel_size=1, padding=0, stride=1,
            norm_layer=nn.BatchNorm2d, eps=1e-5
        )
        self.act = act_layer()

    def forward(self, x):
        x = self.ffn_fc1(x)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x
