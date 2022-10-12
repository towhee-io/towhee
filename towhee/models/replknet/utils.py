# Pytorch implementation of utils for RepLKNet model in the paper:
#       "Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs"
#       (https://arxiv.org/abs/2203.06717)
#
# Inspired by the original code from https://github.com/DingXiaoH/RepLKNet-pytorch
#
# All additions & modifications by Copyright 2021 Zilliz. All rights reserved.
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

from torch import nn
from towhee.models.layers.droppath import DropPath
from towhee.models.layers.conv_bn_activation import Conv2dBNActivation


def fuse_bn(conv: nn.Module, bn: nn.Module):
    """
    Fuse conv & bn modules.
    """
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ConvFFN(nn.Module):
    """
    Convolutional FFN module.

    Args:
        channels (`int`): input dimension (same as output dimension)
        internal_channels (`int`): hidden dimension
        drop_rate (`float`): drop rate of drop path
    """
    def __init__(self, channels, internal_channels,  drop_rate):
        super().__init__()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.preffn_bn = nn.BatchNorm2d(channels)
        self.pw1 = Conv2dBNActivation(
            in_planes=channels, out_planes=internal_channels,
            kernel_size=1, stride=1, padding=0, groups=1,
            norm_layer=nn.BatchNorm2d,
        )
        self.pw2 = Conv2dBNActivation(
            in_planes=internal_channels, out_planes=channels,
            kernel_size=1, stride=1, padding=0, groups=1,
            norm_layer=nn.BatchNorm2d
        )
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
