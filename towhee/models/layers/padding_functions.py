# Copyright 2021 Ross Wightman . All rights reserved.
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
import math
from typing import List, Tuple
import torch
import torch.nn.functional as F

# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    """
    Calculate symmetric padding for a convolution
    Args:
        kernel_size(`Int`):
            Convolution kernel size.
        stride(`Int`):
            Convolution stride parameter.
        dilation(`Int`):
            Convolution dilation parameter.
    Returns:
        (`Int`)
            Padding size to keep.
    """
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def get_same_padding(x: int, k: int, s: int, d: int) -> int:
    """
    Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
    Args:
        x(`Int`):
            Input tensor shape.
        k(`Int`):
            Convolution kernel size.
        s(`Int`):
            Convolution stride parameter.
        d(`Int`):
            Convolution dilation parameter.
    Returns:
        (`Int`):
            Padding value for 'SAME' padding.
    """
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> bool:
    """
    Can SAME padding for given args be done statically?
    Args:
        kernel_size(`Int`):
            Convolution kernel size.
        stride(`Int`):
            Convolution stride parameter.
        dilation(`Int`):
            Convolution dilation parameter.
    Returns:
        (`Bool`): whether SAME padding can be done statically.
    """
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def pad_same(x: torch.Tensor, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0) -> torch.Tensor:
    """
    Dynamically pad input x with 'SAME' padding for conv with specified args
    Args:
        x(`torch.Tensor`):
            Input tensor.
        k(`List[Int]`):
            Convolution kernel sizes.
        s(`List[Int]`):
            Convolution stride parameters.
        d(`List[Int]`):
            Convolution dilation parameter.
        value(`Float`):
            Value for padding.
    Returns:
        (`torch.Tensor`):
            Output Tensor for conv with 'SAME' padding.
    """
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def get_padding_value(padding: str, kernel_size: int, **kwargs) -> Tuple[int, bool]:
    """
    Args:
        padding(`Str`):
            Padding type, 'same' or 'valid'.
        kernel_size(`Int`):
            Convolution kernel size.
    Returns:
        (`Int`): padding shape.
        (`Bool`): dynamically padding.
    """
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic
