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

from towhee.engine.operator_context_v2 import OperatorContext, OpStatus
from towhee.dataframe import DataFrame, Variable
from towhee.dag import OperatorRepr
from towhee.engine.thread_pool_task_executor import ThreadPoolTaskExecutor

from towhee.tests.test_util.dataframe_test_util import DfWriter
from towhee.tests import CACHE_PATH


class TestOperatorContextV2(unittest.TestCase):
    """
    Op ctx test
    """

    def setUp(self):
        self._task_exec = ThreadPoolTaskExecutor('op_ctx_test_executor',
                                                 CACHE_PATH)
        self._task_exec.start()

    def tearDown(self):
        if self._task_exec.is_alive():
            self._task_exec.stop()
            self._task_exec.join()

    def _create_op_ctx(self):
        df_in = DataFrame('op_test_in', {'num': {'type': 'int', 'index': 0}})
        df_out = DataFrame('op_test_out', {'sum': {'type': 'int', 'index': 0}})
        dfs = {'op_test_in': df_in, 'op_test_out': df_out}

        op_repr = OperatorRepr(
            name='add_operator',
            function='mock_operators/add_operator',
            init_args={'factor': 5},
            inputs=[
                {'name': 'num', 'df': 'op_test_in', 'col': 0},
            ],
            outputs=[{'df': 'op_test_out'}],
            iter_info={'type': 'map'}
        )

        op_ctx = OperatorContext(op_repr, dfs)
        self.assertEqual(op_ctx.name, 'add_operator')
        self.assertEqual(op_ctx.status, OpStatus.NOT_RUNNING)
        op_ctx.start(self._task_exec)
        self.assertEqual(op_ctx.status, OpStatus.RUNNING)
        return df_in, df_out, op_ctx

    def test_op_ctx(self):
        df_in, df_out, op_ctx = self._create_op_ctx()

        data = (Variable('int', 1), Variable(
            'str', 'test'), Variable('float', 0.1))
        data_size = 20
        t = DfWriter(df_in, data_size, data=data)
        t.set_sealed_when_stop()
        t.start()
        t.join()
        op_ctx.join()
        self.assertEqual(op_ctx.status, OpStatus.FINISHED)
        df_out.seal()
        map_iter = df_out.map_iter()
        for item in map_iter:
            self.assertEqual(item[0][0].value, 6)

    def test_op_ctx_failed(self):
        df_in, df_out, op_ctx = self._create_op_ctx()

        # Set errer data
        data = (Variable('str', 'test'), )
        data_size = 20
        t = DfWriter(df_in, data_size, data=data)
        t.set_sealed_when_stop()
        t.start()
        t.join()
        op_ctx.join()
        self.assertEqual(op_ctx.status, OpStatus.FAILED)
        df_out.seal()

        self.assertEqual(df_out.size, 0)

    def test_op_ctx_stop(self):
        df_in, _, op_ctx = self._create_op_ctx()

        data = (Variable('int', 1), Variable(
            'str', 'test'), Variable('float', 0.1))
        data_size = 20
        t = DfWriter(df_in, data_size, data=data)
        t.start()
        t.join()
        op_ctx.stop()
        op_ctx.join()
        self.assertEqual(op_ctx.status, OpStatus.FINISHED)
