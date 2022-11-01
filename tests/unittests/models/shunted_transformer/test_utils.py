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

from towhee.models.shunted_transformer import OverlapPatchEmbed, Mlp


class TestUtils(unittest.TestCase):
    """
    Test utils for Shunted Transformer
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_patch_embed2d(self):
        test_shape = (1, 3, 4, 4)
        fake_img = torch.rand(test_shape).to(self.device)
        model = OverlapPatchEmbed(img_size=test_shape[-1], patch_size=2, stride=1, embed_dim=5).to(self.device)
        out = model(fake_img)
        self.assertTrue(out.shape == (1, 25, 5))

    def test_mlp(self):
        dummy_input = torch.rand(1, 16, 3).to(self.device)
        model = Mlp(in_features=3, hidden_features=5, out_features=4).to(self.device)
        out = model(dummy_input, 4, 4)
        self.assertTrue(out.shape == (1, 16, 4))


if __name__ == '__main__':
    unittest.main()
