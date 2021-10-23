# Copyright 2021 Ross Wightman and Zilliz. All rights reserved.
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
from torch.nn import functional as F

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)

class GELU(nn.Module):
    """
    GELU activiation layer.

    Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    Described in: https://arxiv.org/abs/1606.08415.

    Args:
        inplace(`Bool`):
            whether use inplace version.
    Returns:
            output tensor after activation.
    """
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        if inplace is True:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)
