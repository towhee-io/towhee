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

from torch import nn
import unittest
from towhee.models.utils.fuse_bn import fuse_bn


class FuseBNTest(unittest.TestCase):
    def test_fuse_bn(self):
        """
        Test Fuse BN function.
        """
        conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        bn = nn.BatchNorm2d(3)
        f = fuse_bn(conv, bn)
        # print(len(f), f[0].shape, f[1].shape)
        self.assertTrue(len(f) == 2)
        self.assertTrue(f[0].shape == (3, 3, 1, 1))
        self.assertTrue(f[1].shape == (3,))

    def test_double(self):
        """
        Test Fuse BN function.
        """
        conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        bn = nn.BatchNorm2d(3)
        f = fuse_bn(conv, bn)
        # print(len(f), f[0].shape, f[1].shape)
        self.assertTrue(len(f) == 2)
        self.assertTrue(f[0].shape == (6, 3, 1, 1))
        self.assertTrue(f[1].shape == (6,))


if __name__ == "__main__":
    unittest.main()
