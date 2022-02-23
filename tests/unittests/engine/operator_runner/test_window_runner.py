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

from towhee.dataframe import DataFrame
from towhee.dataframe.iterators import MapIterator
from towhee.engine.operator_io import create_reader, create_writer
from towhee.engine.operator_runner.runner_base import RunnerStatus
from towhee.engine.operator_runner.window_runner import WindowRunner
from tests.unittests.mock_operators.sum_operator.sum_operator import SumOperator


def run(runner):
    runner.process()


class TestRunner(unittest.TestCase):
    """
    MapRunner test
    """

    def _create_test_obj(self):
        input_df = DataFrame('input', [('num', 'int')])
        out_df = DataFrame('output', [('sum', 'int')])
        writer = create_writer('window', [out_df])
        params = {'batch_size': 5, 'step': 3}
        reader = create_reader(input_df, 'window', {'num': 0}, params)
        runner = WindowRunner('window_test', 0, 'sum_operator', 'main',
                              'mock_operators', {},
                              [reader], writer)
        return input_df, out_df, runner

    def test_window_runner(self):
        df_in, out_df, runner = self._create_test_obj()

        runner.set_op(SumOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        it = MapIterator(df_in, True)
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        for _ in range(100):
            df_in.put({'num': 1})
        df_in.seal()
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FINISHED)
        self.assertEqual(out_df.size, 68)
        out_df.seal()
        it = MapIterator(out_df, True)

        count = 0
        for item in it:
            if count < 64:
                self.assertEqual(item[0][0], 5)
            elif count < 66:
                self.assertEqual(item[0][0], 4)
            else:
                self.assertEqual(item[0][0], 1)
            count += 1

    def test_window_runner_with_error(self):
        df_in, _, runner = self._create_test_obj()
        runner.set_op(SumOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        df_in.put({'num': 'error_data'})
        df_in.seal()
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FAILED)


if __name__ == '__main__':
    unittest.main()
