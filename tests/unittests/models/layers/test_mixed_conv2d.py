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
from towhee.models.layers.mixed_conv2d import MixedConv2d

class MxiedConv2dTest(unittest.TestCase):
    def test_mixed_conv2d(self):
        in_channels = 32
        out_channels = 64
        kernel_size = 3
        stride = 1
        padding = 'same'
        module = MixedConv2d(in_channels, out_channels, kernel_size, stride, padding)
        x = torch.FloatTensor(1,32,224,224)
        output = module(x)
        gt_output_shape = torch.Size((1,64,224,224))
        self.assertTrue(output.shape == gt_output_shape)

