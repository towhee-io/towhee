# Copyright 2022 Zilliz. All rights reserved.
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

import torch
import unittest
from towhee.models.layers.ffn import FFNBlock


class ConvBNActivationTest(unittest.TestCase):
    def test_ffn(self):
        """
        Test FFN block.
        """
        x = torch.randn(1, 3, 1, 3)
        layer1 = FFNBlock(in_channels=3)
        layer2 = FFNBlock(in_channels=3, hidden_channels=2, out_channels=2)
        y1 = layer1(x)
        y2 = layer2(x)
        # print(y1.shape, y2.shape)
        self.assertTrue(y1.shape == torch.Size([1, 3, 1, 3]))
        self.assertTrue(y2.shape == torch.Size([1, 2, 1, 3]))


if __name__ == "__main__":
    unittest.main()
