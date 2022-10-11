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
#
# Original code from https://github.com/Atze00/MoViNet-pytorch
#
# Modified by Zilliz.

from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple, Union

from torch.nn.modules.utils import _triple, _pair
from torch import nn


class Conv2dBNActivation(nn.Sequential):
    """
    Conv2d with BatchNorm & Activation

    Args:
        in_planes (`int`):
            Number of input channels.
        out_planes (`int`):
            Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]):
            Kernel size for conv layer.
        padding (`Union[int, Tuple[int, int]]`):
            Padding for conv layer.
        stride (`Union[int, Tuple[int, int]]`):
            Stride for conv layer.
        norm_layer (`nn.Module`):
            Norm layer.
        eps (`float`):
            The number of eps for norm layer.
        activation_layer (`nn.Module`):
            Activation layer
        **kwargs (`Any`):
            Other parameters.
    """
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            *,
            kernel_size: Union[int, Tuple[int, int]],
            padding: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            eps: float = 0.001,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        padding = _pair(padding)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride
        dict_layers = OrderedDict({
            "conv2d": nn.Conv2d(in_planes, out_planes,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=groups,
                                **kwargs),
            "norm": norm_layer(out_planes, eps=eps),
            "act": activation_layer()
        })

        self.out_channels = out_planes
        super().__init__(dict_layers)


class Conv3DBNActivation(nn.Sequential):
    """
    Conv3d with BatchNorm & Activation

    Args:
        in_planes (`int`):
            Number of input channels.
        out_planes (`int`):
            Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]):
            Kernel size for conv layer.
        padding (`Union[int, Tuple[int, int]]`):
            Padding for conv layer.
        stride (`Union[int, Tuple[int, int]]`):
            Stride for conv layer.
        norm_layer (`nn.Module`):
            Norm layer.
        eps (`float`):
            The number of eps for norm layer.
        activation_layer (`nn.Module`):
            Activation layer
        **kwargs (`Any`):
            Other parameters.
    """
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            *,
            kernel_size: Union[int, Tuple[int, int, int]],
            padding: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = 1,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            eps: float = 0.001,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any,
    ) -> None:
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride
        dict_layers = OrderedDict({
            "conv3d": nn.Conv3d(in_planes, out_planes,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=groups,
                                **kwargs),
            "norm": norm_layer(out_planes, eps=eps),
            "act": activation_layer()
        })

        self.out_channels = out_planes
        super().__init__(dict_layers)
