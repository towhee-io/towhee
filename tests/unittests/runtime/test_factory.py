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

from towhee.runtime.factory import ops, register


class TestOps(unittest.TestCase):
    """
    Test ops
    """
    def test_property(self):
        op = ops.my_namespace.my_operator('a', 'b', c='c')
        self.assertEqual(op.name, 'my-namespace/my-operator')
        self.assertEqual(op.init_args, ('a', 'b'))
        self.assertEqual(op.init_kws, {'c': 'c'})
        self.assertEqual(op.tag, 'main')

        with self.assertRaises(RuntimeError):
            op.get_op()

    def test_revision(self):
        op0 = ops.test_revision().revision()
        self.assertEqual(op0.tag, 'main')
        self.assertEqual(op0()[0], 'main')

        op1 = ops.test_revision().revision('v1')
        self.assertEqual(op1.tag, 'v1')
        self.assertEqual(op1()[0], 'v1')

    def test_latest(self):
        op = ops.test_revision().latest()
        self.assertTrue(op.is_latest)
        self.assertEqual(op()[0], 'main')

    def test_local(self):
        # pylint: disable=protected-access
        op = ops.local.add_operator(10)
        self.assertEqual(op._op, None)
        res = op(2)
        self.assertNotEqual(op._op, None)
        self.assertEqual(op.name, 'local/add-operator')
        self.assertEqual(op.init_args, (10,))
        self.assertEqual(op.init_kws, {})
        self.assertEqual(op.tag, 'main')
        self.assertEqual(res.sum, 12)

    def test_registry(self):
        # pylint: disable=unused-variable
        @register(name='add_operator')
        class AddOperator:
            def __init__(self, factor):
                self.factor = factor

            def __call__(self, x):
                return self.factor + x

        @register
        def sub_operator(x, y):
            return x - y

        op = ops.sub_operator()
        res = op(11, 2)
        self.assertEqual(op.name, 'sub-operator')
        self.assertEqual(op.init_args, ())
        self.assertEqual(op.init_kws, {})
        self.assertEqual(op.tag, 'main')
        self.assertEqual(res, 9)

        op = ops.add_operator(factor=11)
        res = op(2)
        self.assertEqual(op.name, 'add-operator')
        self.assertEqual(op.init_args, ())
        self.assertEqual(op.init_kws, {'factor': 11})
        self.assertEqual(res, 13)
