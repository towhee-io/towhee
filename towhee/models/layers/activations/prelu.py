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

class PReLU(nn.PReLU):
    """
    PReLU activiation layer.

    Applies PReLU (w/ dummy inplace arg).Described in: https://arxiv.org/abs/1502.01852.

    Args:
        inplace(`Bool`):
            whether use inplace version.
    Returns:
        (`torch.Tensor`)
            output tensor after activation.
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False) -> None:
        super().__init__(num_parameters=num_parameters, init=init)
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.weight)

