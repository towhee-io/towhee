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
from torch import nn

from towhee.models.layers.conv_bn_activation import Conv2dBNActivation
from towhee.models.replknet import fuse_bn


class TestUtils(unittest.TestCase):
    """
    Test RepMLP utils
    """
    def test_fuse_bn(self):
        conv_bn = Conv2dBNActivation(
            in_planes=4,
            out_planes=3,
            kernel_size=2,
            stride=1,
            padding=None,
            groups=1,
            dilation=1,
            activation_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d
        )
        conv, bn = conv_bn.conv2d, conv_bn.norm
        k, b = fuse_bn(conv, bn)

        # print(k.shape, b.shape)
        self.assertTrue(k.shape == (3, 4, 2, 2))
        self.assertTrue(b.shape == (3,))


if __name__ == '__main__':
    unittest.main()
