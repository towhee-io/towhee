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

from towhee.models.layers.window_attention3d import WindowAttention3D


class WindowAttention3DTest(unittest.TestCase):
    def test_window_attention3d(self):
        dim = 144
        window_size = [2, 2, 2]
        num_heads = 8
        wa3 = WindowAttention3D(dim=dim, window_size=window_size, num_heads=num_heads)
        num_windows = 2 * 2 * 2
        b = 3
        n = 8
        c = dim
        input_tensor = torch.rand(num_windows * b, n, c)
        out = wa3(input_tensor)  # torch.Size([24, 8, 144])
        self.assertTrue(out.shape == torch.Size([24, 8, 144]))
