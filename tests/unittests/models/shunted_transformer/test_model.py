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

from towhee.models.shunted_transformer import create_model


class TestModel(unittest.TestCase):
    """
    Test Shunted Transformer
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_base(self):
        dummy_img = torch.rand(1, 3, 16, 16).to(self.device)

        model1 = create_model(img_size=16, num_classes=10, num_stages=2,
                              embed_dims=[2, 4], num_heads=[2, 4], mlp_ratios=[2, 2],
                              depths=[2, 3], sr_ratios=[2, 1]).to(self.device)
        outs1 = model1(dummy_img)
        self.assertTrue(outs1.shape == (1, 10))

        model2 = create_model(img_size=16, num_classes=0, num_stages=2,
                              embed_dims=[2, 4], num_heads=[2, 4], mlp_ratios=[2, 2],
                              depths=[2, 3], sr_ratios=[2, 1]).to(self.device)
        outs2 = model2(dummy_img)
        self.assertTrue(outs2.shape == (1, 4))

    def test_model_names(self):
        with self.assertRaises(ValueError) as e:
            create_model(model_name='test')
        self.assertEqual(str(e.exception),
                         'Invalid model name: test.')

        model_b = create_model(model_name='shunted_b', pretrained=False)
        self.assertTrue(model_b.depths == [3, 4, 24, 2])
        self.assertTrue(model_b.num_conv == 2)
        self.assertTrue(model_b.embed_dims == [64, 128, 256, 512])
        self.assertTrue(model_b.num_heads == [2, 4, 8, 16])
        self.assertTrue(model_b.mlp_ratios == [8, 8, 4, 4])
        self.assertTrue(model_b.sr_ratios == [8, 4, 2, 1])
        self.assertTrue(model_b.num_classes == 1000)
        self.assertTrue(model_b.num_stages == 4)

        model_t = create_model(model_name='shunted_t', pretrained=False)
        self.assertTrue(model_t.depths == [1, 2, 4, 1])
        self.assertTrue(model_t.num_conv == 0)

        model_s = create_model(model_name='shunted_s', pretrained=False)
        self.assertTrue(model_s.depths == [2, 4, 12, 1])
        self.assertTrue(model_s.num_conv == 1)


if __name__ == '__main__':
    unittest.main()
