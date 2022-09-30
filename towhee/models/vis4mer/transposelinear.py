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
import torch
import math

import opt_einsum.contract as contract


class TransposedLinear(nn.Module):
    """
    Linear module on the second-to-last dimension
    Args:
        d_input (str): input dimension
        d_output (int): output dimension
        bias (bool): use bias
    """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output, 1))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = 0.0

    def forward(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias
