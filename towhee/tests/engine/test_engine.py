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
from towhee.engine.operator_context import OperatorContext
from towhee.dataframe import DataFrame, Variable


class TestEngine(unittest.TestCase):
    """
    combine tests of engine/scheduler/task-executor/task
    """

    def setUp(self):
        graph_ctx = GraphContext()
        self._df_in = DataFrame('op_test_in')
        self._df_out = DataFrame('op_test_out')

        op = OperatorContext({
            'name': 'mock_operators/add_operator',
            'op_args': {'factor': 2},
            'iter_type': 'Map',
            'inputs_index': {
                'num': 0
            }
        }, [self._df_in], [self._df_out])

        self._df_in.put((Variable('int', 1), ))

        # TODO (junjie.jiangjjj) use repr to create ctx
        graph_ctx._op_contexts = [op]  # pylint: disable=protected-access

        self._pipeline = Pipeline(graph_ctx, [self._df_in, self._df_out])

    def test_engine(self):
        conf = EngineConfig()
        conf.cache_path = CACHE_PATH
        conf.sched_interval_ms = 20
        engine = Engine()
        engine.add_pipeline(self._pipeline)
        engine.start()
        time.sleep(0.1)
        _, ret = self._df_out.get(0, 1)
        self.assertEqual(ret[0], {'sum': 3})

        self._df_in.put((Variable('int', 3), ))
        time.sleep(0.1)
        _, ret = self._df_out.get(1, 1)
        self.assertEqual(ret[0], {'sum': 5})
        engine.stop()
