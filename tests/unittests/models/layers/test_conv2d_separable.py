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
import torch

from towhee.models.layers.conv2d_separable import SeparableConv2d


class TestOperator(unittest.TestCase):
    """
    Test Separable Conv2D
    """
    x = torch.rand(1, 2, 4, 4)

    def test_separable_conv2d(self):
        conv = SeparableConv2d(
            in_c=2, out_c=3, kernel_size=2, stride=(2, 2), in_f=4, in_t=4,
        )
        outs = conv(self.x)
        self.assertTrue(outs.shape == (1, 3, 2, 2))

    def test_fuller(self):
        conv = SeparableConv2d(
            in_c=2, out_c=3, kernel_size=2, stride=(2, 2), in_f=4, in_t=4, fuller=True, relu_after_bn=True
        )
        conv.hack()
        outs = conv(self.x)
        self.assertTrue(outs.shape == (1, 3, 2, 2))


if __name__ == '__main__':
    unittest.main()
