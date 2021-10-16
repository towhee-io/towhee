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

from towhee.engine.engine import Engine, EngineConfig
from towhee.tests import CACHE_PATH
from towhee.engine.pipeline import Pipeline
from towhee.dataframe import DataFrame, Variable
from towhee.tests.test_util import SIMPLE_PIPELINE_YAML
from towhee.dag import OperatorRepr


class TestEngine(unittest.TestCase):
    """
      combine tests of engine/scheduler/task-executor/task
      """

    def setUp(self):
        conf = EngineConfig()
        conf.cache_path = CACHE_PATH
        conf.sched_interval_ms = 20
        engine = Engine()
        if not engine.is_alive():
            engine.start()

    def test_engine(self):

        start_op_repr = OperatorRepr(
            name='_start_op',
            function='internal',
            init_args={},
            inputs=[
                {'name': 'num', 'df': 'op_test_in', 'col': 0}
            ],
            outputs=[{'df': 'op_test_in'}],
            iter_info={'type': 'map'}
        )

        add_op_repr = OperatorRepr(
            name='mock_operators/add_operator',
            function='mock_operators/add_operator',
            init_args={'factor': 2},
            inputs=[
                {'name': 'num', 'df': 'op_test_in', 'col': 0}
            ],
            outputs=[{'df': 'op_test_out'}],
            iter_info={'type': 'map'}
        )

        end_op_repr = OperatorRepr(
            name='_end_op',
            function='internal',
            init_args={},
            inputs=[
                {'name': 'sum', 'df': 'op_test_out', 'col': 0}
            ],
            outputs=[{'df': 'op_test_out'}],
            iter_info={'type': 'map'}
        )

        df_in_repr = DataFrameRepr(
            name='op_test_in',
            columns=[VariableRepr('num', 'int')]
        )

        df_out_repr = DataFrameRepr(
            name='op_test_out',
            columns=[VariableRepr('sum', 'int')]
        )

        op_reprs = {
            start_op_repr.name: start_op_repr,
            add_op_repr.name: add_op_repr,
            end_op_repr.name: end_op_repr
        }

        df_reprs = {
            df_in_repr.name: df_in_repr,
            df_out_repr.name: df_out_repr
        }

        graph_repr = GraphRepr('add', op_reprs, df_reprs)

        df_in = DataFrame(
            'op_test_in', {'sum': {'index': 0, 'type': 'int'}})
        df_in.put((Variable('int', 1), ))
        df_in.seal()

        self._pipeline = Pipeline(graph_repr)
        engine = Engine()
        engine.add_pipeline(self._pipeline)
        result = self._pipeline(df_in)
        ret = result.get(0, 1)
        self.assertEqual(ret[0][0].value, 3)

        df_in = DataFrame(
            'op_test_in', {'sum': {'index': 0, 'type': 'int'}})
        df_in.put((Variable('int', 3), ))
        df_in.seal()

        result = self._pipeline(df_in)
        ret = result.get(0, 1)
        self.assertEqual(ret[0][0].value, 5)

    def test_simple_pipeline(self):
        with open(SIMPLE_PIPELINE_YAML, 'r', encoding='utf-8') as f:
            p = Pipeline(f.read())
        engine = Engine()
        engine.add_pipeline(p)

        df_in = DataFrame(
            'inputs', {'sum': {'index': 0, 'type': 'int'}})

        df_in.put((Variable('int', 3), ))
        df_in.seal()
        result = p(df_in)
        ret = result.get(0, 1)
        self.assertEqual(ret[0][0].value, 6)

        df_in = DataFrame(
            'inputs', {'sum': {'index': 0, 'type': 'int'}})

        df_in.put((Variable('int', 7), ))
        df_in.seal()
        result = p(df_in)
        ret = result.get(0, 1)
        self.assertEqual(ret[0][0].value, 10)


if __name__ == '__main__':
    unittest.main()
