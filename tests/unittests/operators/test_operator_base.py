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

import os
import unittest

from pathlib import Path

from tests.unittests.mock_operators import ADD_OPERATOR_PATH, load_local_operator
from towhee.operator import SharedType, PyOperator, NNOperator, Operator

_MOCK_OPERATOR_DIR = os.path.join(Path(__file__).parent.parent, 'mock_operators')
NN_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'nn_operator')
PY_OPERATOR_PATH = os.path.join(_MOCK_OPERATOR_DIR, 'py_operator')


class TestOperator(unittest.TestCase):
    """
    Simple operator test
    """
    def test_nnoperator(self):
        self.assertTrue(issubclass(NNOperator, Operator))

        nn_operator = load_local_operator('nn_operator', NN_OPERATOR_PATH)
        pt_op = nn_operator.TestNNOperator()
        tf_op = nn_operator.TestNNOperator('tensorflow')
        self.assertIsInstance(pt_op, NNOperator)
        self.assertIsInstance(tf_op, NNOperator)
        self.assertEqual(pt_op.framework, 'pytorch')
        self.assertEqual(tf_op.framework, 'tensorflow')

        pt_op.framework = 'tensorflow'
        self.assertEqual(pt_op.framework, tf_op.framework)

    def test_pyoperator(self):
        self.assertTrue(issubclass(PyOperator, Operator))

        py_operator = load_local_operator('py_operator', PY_OPERATOR_PATH)
        op = py_operator.TestPyOperator()

        self.assertIsInstance(op, PyOperator)

    def test_add_function(self):
        add_operator = load_local_operator('add_operator', ADD_OPERATOR_PATH)
        op = add_operator.AddOperator(1)
        self.assertTrue(isinstance(op, PyOperator))
        self.assertEqual(op(1).sum, 2)
        self.assertEqual(op.shared_type, SharedType.Shareable)

        op2 = add_operator.AddOperator(3)
        self.assertEqual(op2(1).sum, 4)
        self.assertEqual(op2.shared_type, SharedType.Shareable)


if __name__ == '__main__':
    unittest.main()
