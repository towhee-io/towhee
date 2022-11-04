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


if __name__ == '__main__':
    unittest.main()
