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

from towhee.models.replknet import create_model


class TestModel(unittest.TestCase):
    """
    Test RepMLP model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_model(self):
        x = torch.rand(1, 3, 4, 4).to(self.device)

        model1 = create_model(
            large_kernel_sizes=[13, 7], layers=[1, 2], channels=[7, 5],
            drop_rate=0.2, small_kernel=3, dw_ratio=1, ffn_ratio=4,
            in_channels=3, num_classes=10, deep_fuse=False
        ).to(self.device)
        outs1 = model1(x)
        self.assertTrue(outs1.shape, (1, 10))

        model1.structural_reparam()
        outs1 = model1(x)
        self.assertTrue(outs1.shape, (1, 10))

        model2 = create_model(
            large_kernel_sizes=[13, 7], layers=[1, 2], channels=[7, 5],
            drop_rate=0.2, small_kernel=3, dw_ratio=1, ffn_ratio=4,
            in_channels=3, num_classes=None, out_indices=[1], norm_intermediate_features=True
        ).to(self.device)
        outs2 = model2(x)
        self.assertTrue(len(outs2) == 1)
        self.assertTrue(outs2[0].shape, (1, 10))

        # Test exceptions
        with self.assertRaises(ValueError) as e:
            create_model(num_classes=10, norm_intermediate_features=True)
        self.assertEqual(str(e.exception),
                         'for pretraining, no need to normalize the intermediate feature maps')

        with self.assertRaises(ValueError) as e:
            create_model(num_classes=None)
        self.assertEqual(str(e.exception),
                         'must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')

        with self.assertRaises(ValueError) as e:
            create_model(num_classes=10, out_indices=[0])
        self.assertEqual(str(e.exception),
                         'cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')

    def test_model_name(self):
        replk_b = create_model(model_name='replk_31b_1k')
        self.assertTrue(replk_b.large_kernel_sizes == [31, 29, 27, 13])
        self.assertTrue(replk_b.channels == [128, 256, 512, 1024])
        self.assertTrue(replk_b.num_classes == 1000)

        replk_b_22k = create_model(model_name='replk_31b_22k')
        self.assertTrue(replk_b_22k.large_kernel_sizes == [31, 29, 27, 13])
        self.assertTrue(replk_b_22k.channels == [128, 256, 512, 1024])
        self.assertTrue(replk_b_22k.num_classes == 21841)

        replk_l = create_model(model_name='replk_31l_1k')
        self.assertTrue(replk_l.large_kernel_sizes == [31, 29, 27, 13])
        self.assertTrue(replk_l.channels == [192, 384, 768, 1536])
        self.assertTrue(replk_l.num_classes == 1000)

        replk_l_22k = create_model(model_name='replk_31l_22k')
        self.assertTrue(replk_l_22k.large_kernel_sizes == [31, 29, 27, 13])
        self.assertTrue(replk_l_22k.channels == [192, 384, 768, 1536])
        self.assertTrue(replk_l_22k.num_classes == 21841)

        replk_xl = create_model(model_name='replk_xl')
        self.assertTrue(replk_xl.large_kernel_sizes == [27, 27, 27, 13])
        self.assertTrue(replk_xl.channels == [256, 512, 1024, 2048])
        self.assertTrue(replk_xl.num_classes == 1000)
        self.assertTrue(replk_xl.dw_ratio == 1.5)
        self.assertIsNone(replk_xl.small_kernel)


if __name__ == '__main__':
    unittest.main()
