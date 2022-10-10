# Built on top of the original implementation at https://github.com/md-mohaiminul/ViS4mer
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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
from transposelinear import TransposedLinear
from activation import Activation
from get_initializer import get_initializer


def LinearActivation(
        d_input,
        d_output,
        bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False,
        weight_norm=False,
        **kwargs):
    """
    generate activation module
    Args:
        d_input (int): input dimension
        d_output (int): output dimension
        bias (bool): bias
        zero_bias_init (bool): bias
        transposed (bool): transposed
        initializer (str): initializer
        activation (str): activation
        activate (str): activation
        weight_norm (bool): weight normalization
    return: a linear nn.Module with control over axes order, initialization, and activation
    """

    # Construct core module
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu':
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear
