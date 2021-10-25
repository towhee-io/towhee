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

from towhee.models.layers.droppath import DropPath


class DropPathTest(unittest.TestCase):
    def test_droppath(self):
        mod = DropPath(drop_prob = 0.5)

        h = 224
        w = 224
        in_features = 3

        fake_input = torch.rand(1, in_features, h, w)
        output = mod(fake_input)
        out_features = 3
        gt_output_shape = torch.Size([1, out_features, h, w])
        self.assertTrue(output.shape == gt_output_shape)

