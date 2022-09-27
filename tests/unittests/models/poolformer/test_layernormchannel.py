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
from towhee.models.poolformer.layernormchannel import LayerNormChannel


class LayerNormChannelTest(unittest.TestCase):
    model = LayerNormChannel(3)

    def test_model(self):
        dummy_img = torch.rand(1, 3, 224, 224)
        out = self.model(dummy_img)
        self.assertEqual(out.shape, (1, 3, 224, 224))


if __name__ == '__main__':
    unittest.main()
