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

import doctest
import unittest
from pathlib import Path

from towhee.operator import Operator
from towhee.runtime.operator_manager import OperatorPool
from towhee.runtime.operator_manager import operator_registry


def load_tests(loader, tests, ignore):
    #pylint: disable=unused-argument
    tests.addTests(doctest.DocTestSuite(operator_registry))
    return tests


class TestOperatorPool(unittest.TestCase):
    """
    Basic test case for `OperatorPool`.
    """

    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._op_pool = OperatorPool(cache_path=cache_path)

    def test_acquire_release(self):
        hub_op_id = 'local/add_operator'
        op = self._op_pool.acquire_op(hub_op_id, [], {'factor': 0}, 'main')

        # Perform some simple operations.
        self.assertTrue(isinstance(op, Operator))
        self.assertEqual(op(1).sum, 1)

        # Release and re-acquire the operator.
        self._op_pool.release_op(op)
        op = self._op_pool.acquire_op(hub_op_id, [], {'factor': 0}, 'main')
        # Perform more operations.
        self.assertEqual(op(-1).sum, -1)
        self.assertEqual(op(100).sum, 100)

    def test_shareable_pool(self):
        # Shareable operator only need one
        self._op_pool.clear()
        hub_op_id = 'local/add_operator'
        op1 = self._op_pool.acquire_op(hub_op_id, [], {'factor': 0}, 'main')
        op2 = self._op_pool.acquire_op(hub_op_id, [], {'factor': 0}, 'main')

        self.assertEqual(len(self._op_pool), 1)
        self._op_pool.release_op(op1)
        self.assertEqual(len(self._op_pool), 1)
        self._op_pool.release_op(op2)
        self.assertEqual(len(self._op_pool), 1)

    def test_not_shareable_pool(self):
        self._op_pool.clear()
        hub_op_id = 'local/generator_operator'

        op1 = self._op_pool.acquire_op(hub_op_id, [], {}, 'main')
        op2 = self._op_pool.acquire_op(hub_op_id, [], {}, 'main')

        self.assertEqual(len(self._op_pool), 0)
        self._op_pool.release_op(op1)
        self.assertEqual(len(self._op_pool), 1)
        self._op_pool.release_op(op2)
        self.assertEqual(len(self._op_pool), 2)
        op3 = self._op_pool.acquire_op(hub_op_id, [], {}, 'main')
        self.assertEqual(len(self._op_pool), 1)

        op4 = self._op_pool.acquire_op(hub_op_id, [], {}, 'main')
        op5 = self._op_pool.acquire_op(hub_op_id, [], {}, 'main')
        self._op_pool.release_op(op3)
        self._op_pool.release_op(op4)
        self._op_pool.release_op(op5)
        self.assertEqual(len(self._op_pool), 3)

    def test_not_reusable(self):
        self._op_pool.clear()
        hub_op_id = 'local/flat_operator'

        op1 = self._op_pool.acquire_op(hub_op_id, [], {}, 'main')
        op2 = self._op_pool.acquire_op(hub_op_id, [], {}, 'main')
        self.assertEqual(len(self._op_pool), 0)
        self._op_pool.release_op(op2)
        self.assertEqual(len(self._op_pool), 0)

        self._op_pool.release_op(op1)
        self.assertEqual(len(self._op_pool), 0)

