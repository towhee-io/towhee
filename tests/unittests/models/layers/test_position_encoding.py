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

from towhee.models.layers.position_encoding import build_position_encoding


class TestPositionEncoding(unittest.TestCase):
    """
    Test Transformer Encoder
    """
    x = torch.rand(1, 2)

    def test_sine(self):
        pos_embed = build_position_encoding(hidden_dim=2*2, max_len=4, position_embedding='sine')
        out = pos_embed(self.x)
        self.assertTrue(out.shape == (1, 1, 2))

    def test_learned(self):
        pos_embed = build_position_encoding(hidden_dim=2*2, position_embedding='learned')
        out = pos_embed(self.x)
        self.assertTrue(out.shape == (1, 4, 1, 2))


if __name__ == '__main__':
    unittest.main()
