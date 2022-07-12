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
from towhee.models.max_vit.max_vit_utils import grid_partition, window_partition, \
     window_reverse, grid_reverse


class MaxVitUtilsTest(unittest.TestCase):

    def test_grid_partition_grid_reverse(self):

        x = torch.randn(1, 3, 28, 28)
        out = grid_partition(x)
        self.assertEqual(out.shape, (16, 7, 7, 3))
        origin = grid_reverse(out, (28, 28))
        self.assertEqual(origin.shape, (1, 3, 28, 28))

    def test_window_partitionn(self):

        x = torch.randn(1, 3, 28, 28)
        out = window_partition(x)
        self.assertEqual(out.shape, (16, 7, 7, 3))

    def test_window_reverse(self):
        x = torch.randn(16, 7, 7, 3)
        out = window_reverse(x, (28, 28))
        self.assertEqual(out.shape, (1, 3, 28, 28))


if __name__ == '__main__':
    unittest.main()
