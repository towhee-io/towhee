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
from towhee.models.replknet import fuse_bn, ConvFFN, ReparamLargeKernelConv, RepLKBlock, RepLKNetStage


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

    def test_reparam_large_kernel_conv(self):
        x = torch.rand(1, 5, 3, 3).to(self.device)

        layer0 = ReparamLargeKernelConv(
            in_channels=5, out_channels=3, kernel_size=4, stride=1, groups=1,
            small_kernel=None, small_kernel_merged=False
        ).to(self.device)
        self.assertTrue(hasattr(layer0, 'lkb_origin'))
        self.assertFalse(hasattr(layer0, 'small_conv'))
        outs0 = layer0(x)
        self.assertTrue(outs0.shape == (1, 3, 4, 4))

        layer0.merge_kernel()
        self.assertTrue(hasattr(layer0, 'lkb_reparam'))
        self.assertFalse(hasattr(layer0, 'lkb_origin'))

        layer1 = ReparamLargeKernelConv(
            in_channels=5, out_channels=3, kernel_size=4, stride=1, groups=1,
            small_kernel=2, small_kernel_merged=False
        ).to(self.device)
        self.assertTrue(hasattr(layer1, 'lkb_origin'))
        self.assertTrue(hasattr(layer1, 'small_conv'))
        outs1 = layer1(x)
        self.assertTrue(outs1.shape == (1, 3, 4, 4))

        layer1.merge_kernel()
        self.assertTrue(hasattr(layer1, 'lkb_reparam'))
        self.assertFalse(hasattr(layer1, 'lkb_origin'))
        self.assertFalse(hasattr(layer1, 'small_conv'))

        layer2 = ReparamLargeKernelConv(
            in_channels=5, out_channels=3, kernel_size=4, stride=1, groups=1,
            small_kernel=None, small_kernel_merged=True
        ).to(self.device)
        self.assertTrue(hasattr(layer2, 'lkb_reparam'))
        outs2 = layer2(x)
        self.assertTrue(outs2.shape == (1, 3, 4, 4))

    def test_replk_block(self):
        x = torch.rand(1, 5, 4, 4).to(self.device)

        layer1 = RepLKBlock(
            in_channels=5, dw_channels=2, block_lk_size=7, small_kernel=3, drop_rate=0.
        ).to(self.device)
        outs1 = layer1(x)
        # print(outs1.shape)
        self.assertTrue(outs1.shape == (1, 5, 4, 4))

        layer2 = RepLKBlock(
            in_channels=5, dw_channels=2, block_lk_size=7, small_kernel=3, drop_rate=0.2
        ).to(self.device)
        outs2 = layer2(x)
        # print(outs2.shape)
        self.assertTrue(outs2.shape == (1, 5, 4, 4))

    def test_replknet_stage(self):
        x = torch.rand(1, 5, 4, 4).to(self.device)

        layer1 = RepLKNetStage(
            channels=5, num_blocks=2, stage_lk_size=7,
            drop_rate=[0., 0.1], small_kernel=3, dw_ratio=1, ffn_ratio=4,
            small_kernel_merged=False, norm_intermediate_features=False
        ).to(self.device)
        self.assertTrue(isinstance(layer1.norm, nn.Identity))
        outs1 = layer1(x)
        # print(outs1.shape)
        self.assertTrue(outs1.shape == (1, 5, 4, 4))

        layer2 = RepLKNetStage(
            channels=5, num_blocks=2, stage_lk_size=7,
            drop_rate=0.1, small_kernel=3, dw_ratio=1, ffn_ratio=4,
            small_kernel_merged=True, norm_intermediate_features=True
        ).to(self.device)
        self.assertTrue(isinstance(layer2.norm, nn.BatchNorm2d))
        outs2 = layer2(x)
        # print(outs2.shape)
        self.assertTrue(outs2.shape == (1, 5, 4, 4))


if __name__ == '__main__':
    unittest.main()
