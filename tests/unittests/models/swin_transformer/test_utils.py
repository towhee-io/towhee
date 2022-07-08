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
from towhee.models.swin_transformer.utils import window_reverse, window_partition


class SwinTransformerUtilsTest(unittest.TestCase):

    def test_window_partition(self):
        x = torch.randn(1, 24, 24, 3)
        out = window_partition(x, window_size=4)
        self.assertEqual(out.shape, (36, 4, 4, 3))

    def test_window_reverse(self):
        x = torch.randn(24, 2, 2, 3)
        out = window_reverse(x, window_size=2, h=4, w=4)
        self.assertEqual(out.shape, (6, 4, 4, 3))


if __name__ == '__main__':
    unittest.main()
