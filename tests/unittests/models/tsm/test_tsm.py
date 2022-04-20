# Copyright 2022 Zilliz. All rights reserved.
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
from towhee.models.tsm.tsm import TSN

class TestTSM(unittest.TestCase):
    def test_tsm(self):
        num_class = 256
        num_segments = 1
        modality = 'RGB'
        md = TSN(num_class=num_class, num_segments=num_segments, modality=modality)
        input_tensor = torch.randn(40, 3, 224, 224)
        out = md(input_tensor)
        self.assertTrue(out.shape == torch.Size([40, 256]))

if __name__ == '__main__':
    test=TestTSM()
    test.test_tsm()
    