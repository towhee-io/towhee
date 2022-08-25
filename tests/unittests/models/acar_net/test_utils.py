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

from towhee.models.acar_net.utils import get_bbox_after_aug


class TestUtils(unittest.TestCase):
    """
    Test ACAR-NET Utils
    """
    def test_get_bbox(self):
        bbox = [0, 1, 2, 2]
        net1 = get_bbox_after_aug(aug_info=None, bbox=bbox)
        self.assertTrue(net1 == bbox)

        net2 = get_bbox_after_aug(
            aug_info={'crop_box': [0, 0, 0.5, 0.5], 'pad_ratio': [1, 1], 'flip': True},
            bbox=bbox
        )
        self.assertTrue(net2 is None)

        net3 = get_bbox_after_aug(
            aug_info={'crop_box': [0, 0, 1, 2], 'pad_ratio': [1, 1], 'flip': True},
            bbox=bbox
        )
        self.assertTrue(net3 == [0.0, 0.5, 1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
