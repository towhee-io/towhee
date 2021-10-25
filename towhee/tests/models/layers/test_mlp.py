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

from towhee.models.layers.mlp import Mlp

class MlpTest(unittest.TestCase):
    def test_mlp(self):
        in_features = 32
        hidden_features = 64
        out_features = 128
        mod = Mlp(in_features, hidden_features, out_features)
        fake_input = torch.rand(1, in_features)
        output = mod(fake_input)
        gt_output_shape = torch.Size([1, out_features])
        self.assertTrue(output.shape == gt_output_shape)

