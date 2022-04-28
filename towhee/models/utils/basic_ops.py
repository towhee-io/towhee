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
#
# modified by Zilliz.

import torch

class SegmentConsensus(torch.nn.Module):
    """
    Args:
    """
    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_x):
        self.shape = input_x.size()
        if self.consensus_type == 'avg':
            output = input_x.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_x
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):
    """
    Args:
    """
    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input_x):
        return SegmentConsensus(self.consensus_type, self.dim)(input_x)
