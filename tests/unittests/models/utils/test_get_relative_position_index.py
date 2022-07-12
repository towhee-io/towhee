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
from towhee.models.utils.get_relative_position_index import get_relative_position_index


class GetRelativePositionIndexTest(unittest.TestCase):

    def test_get_relative_position_index(self):
        out = get_relative_position_index(win_h=24, win_w=24)
        self.assertEqual(out.shape, (576, 576))


if __name__ == '__main__':
    unittest.main()
