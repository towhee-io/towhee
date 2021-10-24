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

from towhee.models.layers.sequence_pool import SequencePool


class SequencePoolTest(unittest.TestCase):
    def test_sequence_pool(self):
        model = SequencePool('cls')
        fake_input = torch.rand(8, 10, 10)
        output = model(fake_input)
        self.assertTrue(torch.equal(output, fake_input[:, 0]))
        model = SequencePool('mean')
        output = model(fake_input)
        self.assertTrue(torch.equal(output, fake_input.mean(1)))


if __name__ == '__main__':
    unittest.main()
