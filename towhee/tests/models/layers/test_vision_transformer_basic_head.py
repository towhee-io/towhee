# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import unittest

from towhee.models.layers.vision_transformer_basic_head import VisionTransformerBasicHead


class VisionTransformerBasicHeadTest(unittest.TestCase):
    def test_vision_transformer_basic_head(self):
        batch_size = 8
        seq_len = 10
        input_dim = 10
        out_dim = 20
        head = VisionTransformerBasicHead(
            in_features=input_dim,
            out_features=out_dim,
        )
        fake_input = torch.rand(batch_size, seq_len, input_dim)
        output = head(fake_input)
        gt_shape = (batch_size, out_dim)
        self.assertEqual(tuple(output.shape), gt_shape)


if __name__ == '__main__':
    unittest.main()
