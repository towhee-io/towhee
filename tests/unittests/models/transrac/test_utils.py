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

from towhee.models.transrac import DenseMap


class TestUtils(unittest.TestCase):
    """
    Test TransRAC utils
    """
    # Test DenseMap
    def test_dense_map(self):
        dummy_x = torch.rand(3)
        dense_map = DenseMap(input_dim=3, hidden_dim_1=8, hidden_dim_2=8, out_dim=5)
        out = dense_map(dummy_x)
        self.assertTrue(out.shape == torch.Size([5]))


if __name__ == "__main__":
    unittest.main()
