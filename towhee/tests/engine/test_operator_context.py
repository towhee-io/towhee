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
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

from towhee.engine.operator_context import OperatorContext
from towhee.dataframe import DataFrame, Variable

from towhee.tests.test_util.dataframe_test_util import DfWriter


class TestOperatorContext(unittest.TestCase):
    """UT for `OperatorContext`
    """

    def test_op_context(self):
        df_in = DataFrame('op_test_in')
        df_out = DataFrame('op_test_out')
        op = OperatorContext({
            'name': 'mock_op',
            'op_args': {},
            'iter_type': 'Map',
            'inputs_index': {
                'k1': 0, 'k2': 1
            }
        }, [df_in], [df_out])
        self.assertEqual(len(op.pop_ready_tasks()), 0)

        data = (Variable('int', 1), Variable(
            'str', 'test'), Variable('float', 0.1))
        data_size = 20
        t = DfWriter(df_in, data_size, data=data)
        t.set_sealed_when_stop()
        t.start()
        t.join()

        count = 0
        while True:
            tasks = op.pop_ready_tasks(2)
            if op.finished():
                break
            self.assertEqual(len(tasks), 2)
            count += len(tasks)
            self.assertEqual(op.num_ready_tasks, data_size - count)

        self.assertEqual(count, data_size)

    def test_op_context_multithread(self):
        df_in = DataFrame('op_test_in')
        df_out = DataFrame('op_test_out')
        op = OperatorContext({
            'name': 'mock_op',
            'op_args': {},
            'iter_type': 'Map',
            'inputs_index': {
                'k1': 0, 'k2': 1
            }
        }, [df_in], [df_out])
        self.assertEqual(len(op.pop_ready_tasks()), 0)

        data = (Variable('int', 1), Variable(
            'str', 'test'), Variable('float', 0.1))
        data_size = 20
        t = DfWriter(df_in, data_size, data=data)
        t.set_sealed_when_stop()
        t.start()

        count = 0
        while True:
            tasks = op.pop_ready_tasks(5)
            if op.finished():
                break
            self.assertLessEqual(len(tasks), 2)
            count += len(tasks)

        self.assertEqual(count, data_size)
