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

import torch
import unittest
from towhee.models.layers.conv_bn_activation import Conv2dBNActivation, Conv3DBNActivation

class ConvBNActivationTest(unittest.TestCase):
    def test_conv2d_bn_activation(self):
        """
        Test conv2d bn activation  layer.
        """
        x=torch.randn(1,3,4,4)
        model = Conv2dBNActivation(
                                   in_planes = 3,
                                   out_planes = 3,
                                   kernel_size = (2, 2),
                                   padding = (1, 1),
                                   stride = (1, 1),
                                   )
        y=model(x)
        print(y.shape)
        self.assertTrue(y.shape == torch.Size([1,3,5,5]))

    def test_conv3d_bn_activation(self):
        """
        Test conv2d bn activation  layer.
        """
        x=torch.randn(1,3,3,4,4)
        model = Conv3DBNActivation(
                                   in_planes = 3,
                                   out_planes = 3,
                                   kernel_size = (2, 2, 2),
                                   padding = (1, 1, 1),
                                   stride = (1, 1, 1),
                                   )
        y=model(x)
        print(y.shape)
        self.assertTrue(y.shape == torch.Size([1, 3, 4, 5, 5]))
