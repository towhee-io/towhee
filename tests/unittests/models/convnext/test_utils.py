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
from towhee.models.convnext.utils import LayerNorm, Block


class TestUtils(unittest.TestCase):
    """
    Test utils for ConvNeXt
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_norm_layer(self):
        data1 = torch.rand(1, 2, 2, 3).to(self.device)
        layer1 = LayerNorm(3, data_format='channels_last').to(self.device)
        outs1 = layer1(data1)
        self.assertTrue(outs1.shape == (1, 2, 2, 3))

        data2 = torch.rand(1, 3, 2, 2).to(self.device)
        layer2 = LayerNorm(3, data_format='channels_first').to(self.device)
        outs2 = layer2(data2)
        self.assertTrue(outs2.shape == (1, 3, 2, 2))

    def test_block(self):
        data = torch.rand(1, 3, 2, 2).to(self.device)
        layer1 = Block(dim=3, drop_path=0.).to(self.device)
        outs1 = layer1(data)
        self.assertTrue(outs1.shape == (1, 3, 2, 2))

        layer2 = Block(dim=3, drop_path=0.3).to(self.device)
        outs2 = layer2(data)
        self.assertTrue(outs2.shape == (1, 3, 2, 2))


if __name__ == '__main__':
    unittest.main()
