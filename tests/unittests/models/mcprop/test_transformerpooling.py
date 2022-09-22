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
import os
import torch

from towhee.models.mcprop.transformerpooling import TransformerPooling


class TransformerPoolingTest(unittest.TestCase):
    model = TransformerPooling()

    def test_model(self):
        dummyinput = torch.randn(1024, 4, 1024)
        mask = torch.randint(0, 2, (1024, 4))
        out = self.model.forward(dummyinput, mask=mask)
        self.assertEqual(out.shape, (1024, 1024))


if __name__ == '__main__':
    unittest.main()
