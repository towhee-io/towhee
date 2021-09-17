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


from towhee.tests.mock_operators import ADD_OPERATOR_PATH, load_local_operator
from towhee.operator import SharedType


class TestOperator(unittest.TestCase):
    """
    Simple operator test
    """

    def test_func_operator(self):
        add_operator = load_local_operator(
            'add_operator', ADD_OPERATOR_PATH)
        op = add_operator.AddOperator(1)
        self.assertEqual(op(1).sum, 2)
        self.assertEqual(op.shared_type, SharedType.Shareable)

        op2 = add_operator.AddOperator(3)
        self.assertEqual(op2(1).sum, 4)
        self.assertEqual(op2.shared_type, SharedType.Shareable)


if __name__ == '__main__':
    unittest.main()