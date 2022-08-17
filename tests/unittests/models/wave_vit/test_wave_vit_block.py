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
import unittest
import torch
import numpy as np
from towhee.models.wave_vit.wave_vit_block import rand_bbox, Stem, DownSamples, Block, ClassBlock


class TestWaveVitBlock(unittest.TestCase):
    """
    Test WaveVitBlock model
    """

    def test_rand_bbox(self):
        beta = 1.0
        pooling_scale = 8
        lam = np.random.beta(beta, beta)
        x = torch.randn(1, 24, 24, 3)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=pooling_scale)
        self.assertTrue(bbx1.dtype, np.int64)
        self.assertTrue(bby1.dtype, np.int64)
        self.assertTrue(bbx2.dtype, np.int64)
        self.assertTrue(bby2.dtype, np.int64)

    def test_Stem(self):
        image = torch.randn(1, 3, 24, 24)
        patch_embed = Stem(3, 20, 10)
        x, _, _ = patch_embed(image)
        self.assertTrue(x.shape, (1, 36, 10))

    def test_DownSamples(self):
        xx = torch.randn(5, 10, 3, 3)
        patch_embed = DownSamples(10, 5)
        x, _, _ = patch_embed(xx)
        self.assertTrue(x.shape, (5, 4, 5))

    def test_Block(self):
        data = torch.randn(1, 16, 64)
        model = Block(
            dim=64,
            num_heads=2,
            mlp_ratio=8, block_type="wave")
        model.float()
        out = model(data, 4, 4)
        self.assertTrue(out.shape, (1, 16, 64))
        model2 = Block(
            dim=64,
            num_heads=2,
            mlp_ratio=8, block_type="std_att")
        model2.float()
        out2 = model2(data, 4, 4)
        self.assertTrue(out2.shape, (1, 16, 64))

    def test_ClassBlock(self):
        data = torch.randn(1, 50, 64)
        model = ClassBlock(
                dim=64,
                num_heads=8,
                mlp_ratio=4)
        model.float()
        out = model(data)
        self.assertTrue(out.shape, (1, 50, 64))


if __name__ == "__main__":
    unittest.main()
