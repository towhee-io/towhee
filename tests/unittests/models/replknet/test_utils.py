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
from torch import nn

from towhee.models.layers.conv_bn_activation import Conv2dBNActivation
from towhee.models.replknet import fuse_bn, ConvFFN


class TestUtils(unittest.TestCase):
    """
    Test RepMLP utils
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def test_convffn(self):
        dummy_input = torch.rand(1, 3, 2, 2).to(self.device)
        layer1 = ConvFFN(channels=3, internal_channels=4, drop_rate=0.).to(self.device)
        layer2 = ConvFFN(channels=3, internal_channels=4, drop_rate=0.1).to(self.device)

        outs1 = layer1(dummy_input)
        outs2 = layer2(dummy_input)
        # print(outs.shape)
        self.assertTrue(outs1.shape == (1, 3, 2, 2))
        self.assertTrue(outs2.shape == (1, 3, 2, 2))


if __name__ == '__main__':
    unittest.main()
