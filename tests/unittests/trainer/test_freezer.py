# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torchvision
from towhee.trainer.utils.layer_freezer import LayerFreezer


class TestFreezer(unittest.TestCase):
    """
    Test layer freezer
    """
    model = torchvision.models.resnet50(pretrained=True)
    freezer = LayerFreezer(model)

    def test_freezer_status(self):
        res1 = self.freezer.status(-1)
        res2 = self.freezer.status('fc')
        self.assertEquals(res1, res2, ['unfreezed' for _ in range(len(res1))])

    def test_freezer_by_name(self):
        self.freezer.by_names(['conv1', 'fc'])
        res1 = self.freezer.status('fc')
        res2 = self.freezer.status('conv1')
        self.assertEqual(res1 + res2, ['freezed' for _ in range(len(res1 + res2))])

        self.freezer.by_names(['conv1', 'fc'], freeze=False)
        res1 = self.freezer.status('fc')
        res2 = self.freezer.status('conv1')
        self.assertEqual(res1 + res2, ['unfreezed' for _ in range(len(res1 + res2))])

    def test_freezer_by_idx(self):
        self.freezer.by_idx([0, -1])
        res1 = self.freezer.status(0)
        res2 = self.freezer.status(-1)
        self.assertEqual(res1 + res2, ['freezed' for _ in range(len(res1 + res2))])

        self.freezer.by_idx([0, -1], freeze=False)
        res1 = self.freezer.status(0)
        res2 = self.freezer.status(-1)
        self.assertEqual(res1 + res2, ['unfreezed' for _ in range(len(res1 + res2))])


if __name__ == '__main__':
    unittest.main()
