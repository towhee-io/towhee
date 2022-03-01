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
# from shutil import rmtree

from towhee.dag.graph_repr import GraphRepr
from towhee.dag.variable_repr import VariableRepr
from towhee.dag.dataframe_repr import DataFrameRepr
from towhee.dag import OperatorRepr
from towhee.engine.engine import Engine
from towhee.engine.pipeline import Pipeline
from towhee.dataframe import DataFrame
from towhee.hub.file_manager import FileManagerConfig, FileManager

from tests.unittests.test_util import SIMPLE_PIPELINE_YAML
from tests.unittests import CACHE_PATH


class TestEngine(unittest.TestCase):
    """
    combine tests of engine/scheduler/task-executor/task
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

    def test_engine(self):

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

        self._pipeline = Pipeline(graph_repr)
        engine = Engine()
        engine.add_pipeline(self._pipeline)

        df_in = DataFrame(
            'inputs', [('num', 'int')])
        df_in.put((1, ))
        df_in.seal()

        result = self._pipeline(df_in)
        ret = result.get(0, 1)
        self.assertEqual(ret[0][0], 3)

        df_in = DataFrame(
            'inputs', [('num', 'int')])
        df_in.put((3, ))
        df_in.seal()

        result = self._pipeline(df_in)
        ret = result.get(0, 1)
        self.assertEqual(ret[0][0], 5)

    def test_simple_pipeline(self):
        with open(SIMPLE_PIPELINE_YAML, 'r', encoding='utf-8') as f:
            p = Pipeline(f.read())
        engine = Engine()
        engine.add_pipeline(p)

        df_in = DataFrame(
            'inputs', [('num', 'int')])
        df_in.put((3, ))
        df_in.seal()
        result = p(df_in)
        ret = result.get(0, 1)
        self.assertEqual(ret[0][0], 6)

        df_in = DataFrame(
            'inputs', [('num', 'int')])
        df_in.put((4, ))
        df_in.seal()
        result = p(df_in)
        ret = result.get(0, 1)
        self.assertEqual(ret[0][0], 7)


if __name__ == '__main__':
    unittest.main()
