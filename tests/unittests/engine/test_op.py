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
from pathlib import Path

from towhee.operator.base import Operator
from towhee import op

CACHE_PATH = Path(__file__).parent.parent.resolve()


class TestOperator(Operator):
    def __init__(self, x:int):
        self.x = x

    def __call__(self, y:int) -> int:
        return self.x + y


class TestOp(unittest.TestCase):
    """
    Tests `op` functionality.
    """
    def test_class_op(self):
        test_op = op(TestOperator, x=1)
        res = test_op(1)
        self.assertEqual(res, 2)

    def test_repo_op(self):
        test_op = op('towhee/test-operator', x=1)
        res = test_op(1)
        self.assertEqual(res, 2)

    def test_file_op(self):
        path = CACHE_PATH / 'mock_operators/add_operator/add_operator.py'
        test_op = op(str(path), factor=1)
        res, = test_op(1)
        self.assertEqual(res, 2)


if __name__ == '__main__':
    unittest.main()
