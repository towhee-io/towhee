# Copyright 2021  Facebook. All rights reserved.
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
# This code is modified by Zilliz.


from torch import nn

from towhee.models.layers.resnet_basic_3d_module import ResNetBasic3DModule


def create_resnet_basic_3d_module(
    *,
    in_channels,
    out_channels,
    conv_kernel_size=(3, 7, 7),
    conv_stride=(1, 2, 2),
    conv_padding=(1, 3, 3),
    conv_bias=False,
    conv=nn.Conv3d,
    pool=nn.MaxPool3d,
    pool_kernel_size=(1, 3, 3),
    pool_stride=(1, 2, 2),
    pool_padding=(0, 1, 1),
    norm=nn.BatchNorm3d,
    norm_eps=1e-5,
    norm_momentum=0.1,
    activation=nn.ReLU,
) -> nn.Module:
    """
    ResNet basic 3D stem module. It performs spatiotemporal Convolution, BN, and activation
    following by a spatiotemporal pooling.
                                        Conv3d
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                                        Pool3d
    Args:
        in_channels (`int`):
            Input channel size of the convolution.
        out_channels (`int`):
            Output channel size of the convolution.
        conv_kernel_size (`Tuple`):
            Convolutional kernel size(s).
        conv_stride (`Tuple`):
            Convolutional stride size(s).
        conv_padding (`Tuple`):
            Convolutional padding size(s).
        conv_bias (`bool`):
            Convolutional bias. If true, adds a learnable bias to the output.
        conv (`callable`):
            Callable used to build the convolution layer.
        pool (`Callable`):
            A callable that constructs pooling layer, options include: nn.AvgPool3d,
            nn.MaxPool3d, and None (not performing pooling).
        pool_kernel_size (`Tuple`):
            Pooling kernel size(s).
        pool_stride (`Tuple`):
            Pooling stride size(s).
        pool_padding (`Tuple`):
            Pooling padding size(s).
        norm (`Callable`):
            A callable that constructs normalization layer, options
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (`float`):
            Normalization epsilon.
        norm_momentum (`float`):
            Normalization momentum.
        activation (`Callable`):
            A callable that constructs activation layer, options
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing activation).
    """

    assert conv is not None

    conv_module = conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        bias=conv_bias,
    )
    norm_module = (
        None
        if norm is None
        else norm(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = (
        None
        if activation is None
        else activation()
    )
    pool_module = (
        None
        if pool is None
        else pool(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )
    )
    return ResNetBasic3DModule(
        conv=conv_module,
        norm=norm_module,
        activation=activation_module,
        pool=pool_module,
        )
