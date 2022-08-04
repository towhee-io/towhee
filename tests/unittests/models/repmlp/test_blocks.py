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

from towhee.models.repmlp import GlobalPerceptron


class TestBlocks(unittest.TestCase):
    """
    Test RepMLP blocks
    """
    def test_global_perception(self):
        data = torch.rand(3, 1, 1)
        layer = GlobalPerceptron(input_channels=3, internal_neurons=4)
        out = layer(data)
        # print(out.shape)
        self.assertTrue(out.shape == (1, 3, 1, 1))


if __name__ == '__main__':
    unittest.main()
