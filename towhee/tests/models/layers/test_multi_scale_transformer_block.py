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

from towhee.models.layers.multi_scale_transformer_block import MultiScaleBlock


class MultiScaleTransformerBlockTest(unittest.TestCase):
    def test_multi_scale_transformer_block(self):
        # Change of output dimension.
        block = MultiScaleBlock(10, 20, 2)
        seq_len = 21
        c_dim = 10
        batch_dim = 8
        fake_input = torch.rand(batch_dim, seq_len, c_dim)
        input_shape = (2, 2, 5)
        output, output_shape = block(fake_input, input_shape)
        gt_shape_tensor = torch.rand(8, seq_len, 20)
        self.assertTrue(output.shape == gt_shape_tensor.shape)
        self.assertTrue(output_shape == input_shape)

        # Test pooling.
        block = MultiScaleBlock(10, 20, 2, stride_q=(2, 2, 1))
        c_dim = 10
        batch_dim = 8
        fake_input = torch.rand(batch_dim, seq_len, c_dim)
        input_shape = [2, 2, 5]
        output, output_shape = block(fake_input, input_shape)
        gt_shape_tensor = torch.rand(8, int((seq_len - 1) / 2 / 2) + 1, 20)
        gt_out_shape = [1, 1, 5]
        self.assertTrue(output.shape == gt_shape_tensor.shape)
        self.assertTrue(output_shape == gt_out_shape)
