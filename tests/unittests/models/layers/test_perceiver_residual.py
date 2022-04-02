# Copyright 2021 Zilliz. All rights reserved.
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


import unittest
import torch

from towhee.models.perceiver.MultiHeadAttention import MultiHeadAttention
from towhee.models.perceiver.Residual import Residual


class PerceiverResidualTest(unittest.TestCase):
    def test_perceiver_residual(self):
        num_q_channels = 10
        num_kv_channels = 5
        num_heads = 5
        dropout = 0.2
        mha = MultiHeadAttention(num_q_channels=num_q_channels, num_kv_channels=num_kv_channels,
                                 num_heads=num_heads, dropout=dropout)
        res = Residual(module=mha, dropout=dropout)
        q_tensor = torch.rand(2, 10, 10)
        kv_tensor = torch.rand(2, 5, 5)
        out = res(q_tensor, kv_tensor)
        self.assertTrue(out.shape == torch.Size([2, 10, 10]))
