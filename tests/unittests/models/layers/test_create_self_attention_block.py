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

from towhee.models.perceiver.create_self_attention_block import create_self_attention_block


class PerceiverCreateSelfAttentionBlockTest(unittest.TestCase):
    def test_perceiver_create_self_attention_block(self):
        num_q_channels = 10
        num_heads = 5
        batch_size = 2
        dropout = 0.2
        layer = create_self_attention_block(num_layers=2, num_channels=num_q_channels, num_heads=num_heads,
                                            dropout=dropout)
        q_tensor = torch.rand(batch_size, num_q_channels, num_q_channels)
        out = layer(q_tensor)  # torch.Size([2, 10, 10])
        self.assertTrue(out.shape == torch.Size([2, 10, 10]))
