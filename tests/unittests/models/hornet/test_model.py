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
from towhee.models.hornet import HorNet, GatedConv, Block


class TestModel(unittest.TestCase):
    """
    Test HorNet model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_model(self):
        data = torch.rand(1, 3, 32, 32).to(self.device)

        model1 = HorNet(
            in_chans=3, num_classes=10,
            depths=(2, 2, 3, 3), base_dim=16, drop_path_rate=0.,
            gnconv=GatedConv, block=Block
        ).to(self.device)
        outs1 = model1(data)
        self.assertTrue(outs1.shape == (1, 10))

        model2 = HorNet(
            in_chans=3, num_classes=10,
            depths=(2, 2, 3, 3), base_dim=16, drop_path_rate=0.,
            gnconv=[GatedConv]*4, block=Block, uniform_init=True
        ).to(self.device)
        outs2 = model2(data)
        self.assertTrue(outs2.shape == (1, 10))


if __name__ == '__main__':
    unittest.main()
