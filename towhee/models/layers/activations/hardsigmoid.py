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

import torch
from torch import nn
from torch.nn import functional as F

def hard_sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class HardSigmoid(nn.Module):
    """
    HardSigmoid activiation layer.

    A layer applies the element-wise hardsigmoid function to the input.
    Described in: https://paperswithcode.com/method/hard-sigmoid.

    Args:
        inplace(`Bool`):
            whether use inplace version.
    Returns:
        (`torch.Tensor`)
            output tensor after activation.
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_sigmoid(x, self.inplace)

