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
import towhee


class TestExceptionSafe(unittest.TestCase):
    """
    Unit test for exception safe.
    """

    def test_safe(self):
        retval = towhee.dc['x']([5, 3, 2, 1, 0]) \
                    .safe() \
                    .runas_op['x', 'x2'](lambda x: x-1) \
                    .runas_op['x2', 'y'](lambda x: 10/x) \
                    .drop_empty() \
                    .runas_op(lambda x: x.y) \
                    .to_list()
        self.assertListEqual(retval, [2.5, 5.0, 10.0, -10.0])


if __name__ == '__main__':
    unittest.main()
