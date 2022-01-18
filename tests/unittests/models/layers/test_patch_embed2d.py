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


class PatchEmbedTest(unittest.TestCase):
    def test_patch_embed(self):
        in_c = 3
        img_height = img_width = 224
        mod = PatchEmbed2D()
        fake_input = torch.rand(1, in_c, img_height, img_width)
        output = mod(fake_input)
        gt_output_shape = torch.Size([1, 196, 768])
        self.assertTrue(output.shape == gt_output_shape)
