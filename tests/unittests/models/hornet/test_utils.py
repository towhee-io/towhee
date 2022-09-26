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
from towhee.models.hornet.utils import GatedConv, GlobalLocalFilter, Block


class TestUtils(unittest.TestCase):
    """
    Test utils for HorNet
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_gnconv(self):
        data = torch.rand(1, 4, 2, 3).to(self.device)
        layer1 = GatedConv(dim=4, order=2, gflayer=None, h=2, w=3, s=1.0).to(self.device)
        outs1 = layer1(data)
        self.assertTrue(outs1.shape == (1, 4, 2, 3))

        layer2 = GatedConv(dim=4, order=2, gflayer=GlobalLocalFilter, h=2, w=3, s=1.0).to(self.device)
        outs2 = layer2(data)
        self.assertTrue(outs2.shape == (1, 4, 2, 3))

    def test_filter(self):
        data = torch.rand(1, 4, 2, 3).to(self.device)
        layer = GlobalLocalFilter(dim=4, h=2, w=3).to(self.device)
        outs = layer(data)
        self.assertTrue(outs.shape == (1, 4, 2, 3))

    def test_block(self):
        data = torch.rand(1, 16, 2, 3).to(self.device)
        layer1 = Block(dim=16, drop_rate=0.).to(self.device)
        outs1 = layer1(data)
        self.assertTrue(outs1.shape == (1, 16, 2, 3))

        layer2 = Block(dim=16, drop_rate=0.3, layer_scale_init_value=0)
        outs2 = layer2(data)
        self.assertTrue(outs2.shape == (1, 16, 2, 3))


if __name__ == '__main__':
    unittest.main()
