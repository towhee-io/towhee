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
from towhee.models.utils.init_vit_weights import init_vit_weights


class InitVitWeightsTest(unittest.TestCase):

    def test_init_vit_weights(self):
        norm = torch.nn.LayerNorm(20)
        ones = torch.ones(20)
        zeros = torch.zeros(20)
        init_vit_weights(norm)
        self.assertTrue((norm.weight == ones).all())
        self.assertTrue((norm.bias == zeros).all())


if __name__ == '__main__':
    unittest.main()
