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

from towhee.models.layers.multi_head_attention import MultiHeadAttention


class MHATest(unittest.TestCase):
    def test_mha_with_lrp(self):
        seq_len = 21
        c_dim = 10
        mod = MultiHeadAttention(c_dim, num_heads=2)
        fake_input = torch.rand(8, seq_len, c_dim)
        out1 = mod(fake_input)
        kwargs = {'alpha': 1}
        # torch.Size([8, 21, 10])
        out2 = mod.relprop(out1, **kwargs)
        self.assertTrue(out2.shape == torch.Size([8, 21, 10]))
