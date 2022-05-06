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
from towhee.models.layers.non_local import _NonLocalBlockND, NONLocalBlock1D, NONLocalBlock2D, NONLocalBlock3D


class TestNonLocal(unittest.TestCase):
    """
    Test NonLocal layers.
    """
    def test_non_local_blocknd(self):
        """
        Test non-local blocknd.
        """
        in_channels = 3
        inter_channels = None
        dimension = 3
        sub_sample = True
        bn_layer = True
        net = _NonLocalBlockND(in_channels=in_channels,
                               inter_channels=inter_channels,
                               dimension=dimension,
                               sub_sample=sub_sample,
                               bn_layer=bn_layer)
        input_tensor = torch.randn(1, 3, 1, 4, 4)
        output_tensor = net(input_tensor)
        self.assertTrue(output_tensor.shape == torch.Size([1, 3, 1, 4, 4]))

    def test_non_local_block1d(self):
        """
        Test non-local block1d.
        """
        in_channels = 1
        inter_channels = None
        sub_sample = True
        bn_layer = True
        net = NONLocalBlock1D(in_channels=in_channels,
                              inter_channels=inter_channels,
                              sub_sample=sub_sample,
                              bn_layer=bn_layer)
        input_tensor = torch.randn(1, 1, 4)
        output_tensor = net(input_tensor)
        self.assertTrue(output_tensor.shape == torch.Size([1, 1, 4]))

    def test_non_local_block2d(self):
        """
        Test non-local block2d.
        """
        in_channels = 2
        inter_channels = None
        sub_sample = True
        bn_layer = True
        net = NONLocalBlock2D(in_channels=in_channels,
                              inter_channels=inter_channels,
                              sub_sample=sub_sample,
                              bn_layer=bn_layer)
        input_tensor = torch.randn(1, 2, 4, 4)
        output_tensor = net(input_tensor)
        self.assertTrue(output_tensor.shape == torch.Size([1, 2, 4, 4]))

    def test_non_local_block3d(self):
        """
        Test non-local block3d.
        """
        in_channels = 3
        inter_channels = None
        sub_sample = True
        bn_layer = True
        net = NONLocalBlock3D(in_channels=in_channels,
                              inter_channels=inter_channels,
                              sub_sample=sub_sample,
                              bn_layer=bn_layer)
        input_tensor = torch.randn(1, 3, 1, 4, 4)
        output_tensor = net(input_tensor)
        self.assertTrue(output_tensor.shape == torch.Size([1, 3, 1, 4, 4]))


if __name__ == '__main__':
    unittest.main()

