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

from towhee.models import acar_net


class TestHead(unittest.TestCase):
    """
    Test ACAR-NET heads
    """
    data = {
        'features': [torch.rand(1, 128, 2, 3, 3), torch.rand(1, 16, 8, 3, 3)],
        'rois': torch.rand(1, 5),
        'num_rois': 1,
        'roi_ids': [0, 1],
        'sizes_before_padding': [[1.0, 1.0]]
        }

    def test_linear(self):
        head = acar_net.head('linear', width=144, num_classes=10)
        out = head(self.data)
        self.assertTrue(out.shape == (1, 10))

    def test_acar(self):
        head = acar_net.head('acar', width=144, num_classes=10)
        out = head(self.data)
        self.assertTrue(out.shape == (1, 10))


if __name__ == '__main__':
    unittest.main()
