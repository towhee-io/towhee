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
import time

from towhee.engine.engine import Engine, EngineConfig
from towhee.tests import CACHE_PATH
from towhee.engine.pipeline import Pipeline
from towhee.engine.graph_context import GraphContext
from towhee.engine.operator_context import OperatorContext, OpInfo
from towhee.dataframe import DataFrame, Variable
from towhee.engine._repr_to_ctx import create_ctxs
from towhee.tests.test_util import SIMPLE_PIPELINE_YAML


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
        self._df_in = DataFrame(
            'op_test_in', {'sum': {'index': 0, 'type': 'int'}})
        self._df_out = DataFrame(
            'op_test_out', {'sum': {'index': 0, 'type': 'int'}})

        op = OperatorContext(OpInfo(**{
            'name': 'mock_operators/add_operator',
            'function': 'mock_operators/add_operator',
            'op_args': {'factor': 2},
            'iter_type': 'Map',
            'inputs_index': {
                'num': 0
            }
        }), [self._df_in], [self._df_out])

        self._df_in.put((Variable('int', 1), ))

        # TODO (junjie.jiangjjj) use repr to create ctx
        graph_ctx = GraphContext([op])
        self._pipeline = Pipeline(graph_ctx, [self._df_in, self._df_out])
        engine = Engine()
        engine.add_pipeline(self._pipeline)
        time.sleep(0.1)
        _, ret = self._df_out.get(0, 1)
        self.assertEqual(ret[0][0].value, 3)

        self._df_in.put((Variable('int', 3), ))
        time.sleep(0.1)
        _, ret = self._df_out.get(1, 1)
        self.assertEqual(ret[0][0].value, 5)

    def test_simple_pipeline(self):
        g_ctx, dataframes = create_ctxs(SIMPLE_PIPELINE_YAML)
        p = Pipeline(g_ctx, dataframes.keys())
        engine = Engine()
        engine.add_pipeline(p)
        dataframes['df1'].put((Variable('int', 3),))
        time.sleep(0.3)
        _, ret = dataframes['df3'].get(0, 1)
        self.assertEqual(ret[0][0].value, 6)
