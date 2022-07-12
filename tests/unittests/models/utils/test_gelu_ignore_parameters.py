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
from torch import nn
from towhee.models.utils.gelu_ignore_parameters import gelu_ignore_parameters


class GetRelativePositionIndexTest(unittest.TestCase):

    def test_get_relative_position_index(self):
        out = gelu_ignore_parameters()
        self.assertTrue(out, nn.GELU())


if __name__ == '__main__':
    unittest.main()
