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


class ReparamLargeKernelConv(nn.Module):
    """
    Large Kernel Convolutions with new parameters.

    Args:
        in_channels (`int`): input dimension
        out_channels (`int`): output dimension
        kernel_size (`int`): kernel size of conv
        stride (`int`): stride of conv
        groups (`int`): groups of conv
        small_kernel (`int`): small kernel size
        small_kernel_merged (`bool`): flag to merge
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups,
                 small_kernel, small_kernel_merged=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2.
        # Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = Conv2dBNActivation(
                in_planes=in_channels, out_planes=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=groups,
                norm_layer=nn.BatchNorm2d
            )
            if small_kernel is not None:
                assert small_kernel <= kernel_size, \
                    'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = Conv2dBNActivation(
                    in_planes=in_channels, out_planes=out_channels,
                    kernel_size=small_kernel, stride=stride, padding=small_kernel//2, groups=groups, dilation=1,
                    norm_layer=nn.BatchNorm2d
                )

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv2d, self.lkb_origin.norm)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv2d, self.small_conv.norm)
            eq_b += small_b
            # add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.lkb_origin.conv2d.in_channels, out_channels=self.lkb_origin.conv2d.out_channels,
            kernel_size=self.lkb_origin.conv2d.kernel_size, stride=self.lkb_origin.conv2d.stride,
            padding=self.lkb_origin.conv2d.padding, dilation=self.lkb_origin.conv2d.dilation,
            groups=self.lkb_origin.conv2d.groups, bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')
