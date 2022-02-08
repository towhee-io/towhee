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
from towhee.models.vggish.torch_vggish import VGG


class TestVGG(unittest.TestCase):
    def test_vgg(self):
        model = VGG()
        test_input = torch.randn(3, 1, 96, 64)
        out = model(test_input)
        self.assertTrue(out.shape == (3, 128))

if __name__ == '__main__':
    unittest.main(verbosity=1)
