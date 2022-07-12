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

from towhee.models.layers.cross_attention import CrossAttention


class CoCaCrossAttentionTest(unittest.TestCase):
    def test_coca_cross_attention(self):
        x = torch.randn(1,3,4)
        model = CrossAttention(dim=4,
                               context_dim=32,
                               dim_head=64,
                               heads=8,
                               norm_context=True,
                              )
        image_tokens = torch.randn(1,2,32)
        y = model(x, image_tokens)
        self.assertTrue(y.shape == torch.Size([1,3,4]))
