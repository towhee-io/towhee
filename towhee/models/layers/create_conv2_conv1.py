# Copyright 2021 Zilliz and Facebook. All rights reserved.
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
from towhee.models.layers.conv2_conv1 import Conv2and1D


def create_conv_2p_1d(
    *,
    # Conv configs.
    in_channels,
    out_channels,
    inner_channels=None,
    conv_xy_first=False,
    kernel_size=(3, 3, 3),
    stride=(2, 2, 2),
    padding=(1, 1, 1),
    bias=False,
    dilation=(1, 1, 1),
    groups=1,
    norm=nn.BatchNorm3d,
    norm_eps=1e-5,
    norm_momentum=0.1,
    activation=nn.ReLU,
) -> nn.Module:
    """
    Create a 2plus1d conv layer. It performs spatiotemporal Convolution, BN, and
    Relu following by a spatiotemporal pooling.
    ::
                        Conv_t (or Conv_xy if conv_xy_first = True)
                                           ↓
                                     Normalization
                                           ↓
                                       Activation
                                           ↓
                        Conv_xy (or Conv_t if conv_xy_first = True)
    Normalization options include: BatchNorm3d and None (no normalization).
    Activation options include: ReLU, Softmax, Sigmoid, and None (no activation).
    Args:
        in_channels (`int`):
            Input channel size of the convolution.
        out_channels (`int`):
            Output channel size of the convolution.
        kernel_size (`tuple`):
            Convolutional kernel size(s).
        stride (`tuple`):
            Convolutional stride size(s).
        padding (`tuple`):
            Convolutional padding size(s).
        bias (`bool`):
            Convolutional bias. If true, adds a learnable bias to the output.
        groups (`int`):
            Number of groups in convolution layers. value >1 is unsupported.
        dilation (`tuple`):
            Dilation value in convolution layers. value >1 is unsupported.
        conv_xy_first (`bool`):
            If True, spatial convolution comes before temporal conv
        norm (`callable`):
            A callable that constructs normalization layer, options
            include nn.BatchNorm3d, None (not performing normalization).
        norm_eps (`float`):
            Normalization epsilon.
        norm_momentum (`float`):
            Normalization momentum.
        activation (`callable`):
            A callable that constructs activation layer, options
            include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not performing activation).
    Returns:
        (`nn.Module`):
            2plus1d conv layer.
    """
    if inner_channels is None:
        inner_channels = out_channels

    assert (
        groups == 1
    ), "Support for groups is not implemented in R2+1 convolution layer"
    assert (
        max(dilation) == 1 and min(dilation) == 1
    ), "Support for dillaiton is not implemented in R2+1 convolution layer"

    conv_t_module = nn.Conv3d(
        in_channels=in_channels if not conv_xy_first else inner_channels,
        out_channels=inner_channels if not conv_xy_first else out_channels,
        kernel_size=(kernel_size[0], 1, 1),
        stride=(stride[0], 1, 1),
        padding=(padding[0], 0, 0),
        bias=bias,
    )
    norm_module = (
        None
        if norm is None
        else norm(num_features=inner_channels, eps=norm_eps, momentum=norm_momentum)
    )
    activation_module = None if activation is None else activation()
    conv_xy_module = nn.Conv3d(
        in_channels=inner_channels if not conv_xy_first else in_channels,
        out_channels=out_channels if not conv_xy_first else inner_channels,
        kernel_size=(1, kernel_size[1], kernel_size[2]),
        stride=(1, stride[1], stride[2]),
        padding=(0, padding[1], padding[2]),
        bias=bias,
    )

    return Conv2and1D(
        conv_t=conv_t_module,
        norm=norm_module,
        activation=activation_module,
        conv_xy=conv_xy_module,
        conv_xy_first=conv_xy_first,
    )
