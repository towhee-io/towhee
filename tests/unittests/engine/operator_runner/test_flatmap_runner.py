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
import threading

from towhee.dataframe import DataFrame, Variable
from towhee.engine.operator_runner.runner_base import RunnerStatus
from towhee.engine.operator_runner.flatmap_runner import FlatMapRunner
from towhee.engine.operator_io import create_reader, create_writer
from tests.unittests.mock_operators.repeat_operator.repeat_operator import RepeatOperator


def run(runner):
    runner.process()


class TestFlatmapRunner(unittest.TestCase):
    """
    MapRunner test
    """

    def _create_test_obj(self):
        input_df = DataFrame('input', [('num', 'int')])
        out_df = DataFrame('output', [('num', 'int')])
        writer = create_writer('map', [out_df])
        reader = create_reader(input_df, 'map', {'num': 0})
        runner = FlatMapRunner('test', 0, 'repeat_operator', 'main',
                               'mock_operators', {'repeat': 3},
                               [reader], writer)
        return input_df, out_df, runner

    def test_flatmap_runner(self):
        input_df, out_df, runner = self._create_test_obj()

        runner.set_op(RepeatOperator(3))
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        input_df.put_dict({'num': 1})
        input_df.put_dict({'num': 2})
        input_df.put_dict({'num': 3})
        input_df.seal()
        t.join()

        runner.join()
        out_df.seal()
        self.assertEqual(runner.status, RunnerStatus.FINISHED)
        self.assertEqual(out_df.size, 9)

    def test_flatmap_runner_with_error(self):
        input_df, _, runner = self._create_test_obj()
        runner.set_op(RepeatOperator(3))
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        input_df.put((Variable('str', 'errdata'), ))
        runner.join()
        self.assertTrue('Input is not int type' in runner.msg)
        self.assertEqual(runner.status, RunnerStatus.FAILED)
