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


from towhee.dag.graph_repr import GraphRepr
from towhee.dag.variable_repr import VariableRepr
from towhee.dag.dataframe_repr import DataFrameRepr
import unittest

from towhee.dataframe.iterators import MapIterator
from towhee.engine.graph_context import GraphContext
from towhee.engine.operator_context import OpStatus
from towhee.dag import OperatorRepr
from towhee.engine.thread_pool_task_executor import ThreadPoolTaskExecutor
from towhee.hub.file_manager import FileManagerConfig, FileManager

from tests.unittests import CACHE_PATH


class TestGraphCtx(unittest.TestCase):
    """
    Graph ctx test
    """

    @classmethod
    def setUpClass(cls):
        new_cache = (CACHE_PATH/'test_cache')
        pipeline_cache = (CACHE_PATH/'test_util')
        operator_cache = (CACHE_PATH/'mock_operators')
        fmc = FileManagerConfig()
        fmc.update_default_cache(new_cache)
        pipelines = list(pipeline_cache.rglob('*.yaml'))
        operators = [f for f in operator_cache.iterdir() if f.is_dir()]
        fmc.cache_local_pipeline(pipelines)
        fmc.cache_local_operator(operators)
        FileManager(fmc)

    def setUp(self):
        self._task_exec = ThreadPoolTaskExecutor('Graph_ctx_test',
                                                 CACHE_PATH)
        self._task_exec.start()

    def tearDown(self):
        if self._task_exec.is_alive():
            self._task_exec.stop()
            self._task_exec.join()

    def _crate_graph(self):

        start_op_repr = OperatorRepr(
            name='_start_op',
            function='_start_op',
            init_args={},
            inputs=[
                {'name': 'num', 'df': '_start_df', 'col': 0}
            ],
            outputs=[{'df': 'op_test_in'}],
            iter_info={'type': 'map'}
        )

        add_op_repr = OperatorRepr(
            name='local/add_operator',
            function='local/add_operator',
            init_args={'factor': 2},
            inputs=[
                {'name': 'num', 'df': 'op_test_in', 'col': 0}
            ],
            outputs=[{'df': 'op_test_out'}],
            iter_info={'type': 'map'}
        )

        end_op_repr = OperatorRepr(
            name='_end_op',
            function='_end_op',
            init_args={},
            inputs=[
                {'name': 'sum', 'df': 'op_test_out', 'col': 0}
            ],
            outputs=[{'df': '_end_df'}],
            iter_info={'type': 'map'}
        )

        df_start_repr = DataFrameRepr(
            name='_start_df',
            columns=[VariableRepr('num', 'int')]
        )

        df_in_repr = DataFrameRepr(
            name='op_test_in',
            columns=[VariableRepr('num', 'int')]
        )

        df_out_repr = DataFrameRepr(
            name='op_test_out',
            columns=[VariableRepr('sum', 'int')]
        )

        df_end_repr = DataFrameRepr(
            name='_end_df',
            columns=[VariableRepr('sum', 'int')]
        )

        op_reprs = {
            start_op_repr.name: start_op_repr,
            add_op_repr.name: add_op_repr,
            end_op_repr.name: end_op_repr
        }

        df_reprs = {
            df_start_repr.name: df_start_repr,
            df_in_repr.name: df_in_repr,
            df_out_repr.name: df_out_repr,
            df_end_repr.name: df_end_repr
        }

        graph_repr = GraphRepr('add', '', op_reprs, df_reprs)
        graph_ctx = GraphContext(0, graph_repr)
        return graph_ctx
        # df_in = DataFrame(
        #     'op_test_in', {'sum': {'index': 0, 'type': 'int'}})
        # df_in.put((Variable('int', 1), ))
        # df_in.seal()

    def test_graph_ctx(self):
        graph_ctx = self._crate_graph()

        for op in graph_ctx.op_ctxs.values():
            op.start(self._task_exec)

        graph_ctx((3, ))

        graph_ctx.join()

        for op in graph_ctx.op_ctxs.values():
            self.assertEqual(op.status, OpStatus.FINISHED)

        it = MapIterator(graph_ctx.outputs, True)
        for data in it:
            self.assertEqual(data[0][0], 5)

    def test_graph_ctx_failed(self):
        graph_ctx = self._crate_graph()

        for op in graph_ctx.op_ctxs.values():
            op.start(self._task_exec)

        # Error input data
        graph_ctx(('x', ))

        graph_ctx.join()

        for name, op in graph_ctx.op_ctxs.items():
            if name not in ['_start_op', '_end_op']:
                self.assertEqual(op.status, OpStatus.FAILED)

        self.assertEqual(graph_ctx.outputs.size, 0)

if __name__ == '__main__':
    unittest.main()
