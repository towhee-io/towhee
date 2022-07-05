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

import unittest

from towhee import serve as fmt


class TestFormatUtils(unittest.TestCase):
    """
    Unit test for format utils.
    """

    def test_intend(self):
        a = 'a = np.array([1,2,3])'
        expected_results = '   a = np.array([1,2,3])'

        self.assertListEqual(fmt.intend([a], 3), [expected_results])


if __name__ == '__main__':
    unittest.main()
