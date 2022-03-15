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

from towhee.models.layers.swin_transformer_block3d import SwinTransformerBlock3D


class SwinTransformerBlock3DTest(unittest.TestCase):
    def test_swin_transformer_block3d(self):
        dim = 144
        window_size = [2, 2, 2]
        num_heads = 8
        stb3d = SwinTransformerBlock3D(dim=dim, num_heads=num_heads, window_size=window_size)
        b = 3
        d = 4
        h = 4
        w = 4
        c = dim
        input_tensor = torch.rand(b, d, h, w, c)
        mask_matrix = None
        # torch.Size([3, 4, 4, 4, 144])
        out = stb3d(input_tensor, mask_matrix)
        self.assertTrue(out.shape == torch.Size([3, 4, 4, 4, 144]))
