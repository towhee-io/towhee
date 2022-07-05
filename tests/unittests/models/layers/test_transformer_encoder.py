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

from towhee.models.layers.transformer_encoder import TransformerEncoder


class TransformerEncoderTest(unittest.TestCase):
    """
    Test Transformer Encoder
    """
    def test_transformer_encoder(self):
        dummy_x = torch.rand(2, 4)
        mode = TransformerEncoder(d_model=4, n_head=1, dim_ff=1, dropout=0.0, num_layers=1, num_frames=2)
        out = mode(dummy_x)
        self.assertTrue(out.shape == (2, 2, 4))


if __name__ == '__main__':
    unittest.main()
