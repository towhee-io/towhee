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

from towhee.models.layers.patch_embed2d import PatchEmbed2D


class PatchEmbed2DTest(unittest.TestCase):
    def test_patch_embed2d(self):
        test_shape1 = (1, 3, 224, 224)
        test_shape2 = (1, 3, 5, 224, 224)
        fake_img = torch.rand(test_shape1)
        fake_video = torch.rand(test_shape2)
        model = PatchEmbed2D()
        out1 = model(fake_img)
        out2 = model(fake_video)
        self.assertTrue(out1.shape == (1, 196, 768))
        self.assertTrue(out2.shape == (5, 196, 768))

    def test_patch_embed2d_with_lrp(self):
        test_shape1 = (1, 3, 224, 224)
        fake_img = torch.rand(test_shape1)
        kwargs = {'alpha': 1}
        model = PatchEmbed2D()
        out1 = model(fake_img)
        # torch.Size([1, 3, 224, 224])
        out2 = model.relprop(out1, **kwargs)
        self.assertTrue(out2.shape == torch.Size([1, 3, 224, 224]))
