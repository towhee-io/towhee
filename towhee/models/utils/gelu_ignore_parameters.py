# Original pytorch implementation by:
# 'MaxViT: Multi-Axis Vision Transformer'
#       - https://arxiv.org/pdf/2204.01697.pdf
# Original code by / Copyright 2021, Christoph Reich.
# Modifications & additions by / Copyright 2022 Zilliz. All rights reserved.
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


def gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    _ = args
    _ = kwargs
    activation = nn.GELU()
    return activation
