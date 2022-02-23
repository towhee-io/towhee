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
from towhee.engine.operator_runner.filter_runner import FilterRunner
from tests.unittests.mock_operators.zero_drop import zero_drop


def run(runner):
    runner.process()


class TestFilterRunner(unittest.TestCase):
    """
    FilterRunner test
    """

    def _create_test_obj(self):
        input_df = DataFrame('input', [('num', 'int')])
        out_df = DataFrame('output', [('num', 'int')])
        writer = create_writer('filter', [out_df])
        reader = create_reader(input_df, 'filter', {'num': 0})
        runner = FilterRunner('test', 0, 'zero_operator', 'main',
                              'mock_operators', {},
                              [reader], writer)
        return input_df, out_df, runner

    def test_filter_runner(self):
        input_df, out_df, runner = self._create_test_obj()
        runner.set_op(zero_drop.ZeroDrop())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        input_df.put({'num': 0})
        input_df.put({'num': 1})
        input_df.put({'num': 2})
        input_df.put({'num': 0})
        input_df.put({'num': 0})
        input_df.put({'num': 3})
        input_df.put({'num': 0})
        input_df.seal()
        runner.join()
        out_df.seal()
        res = 1
        it = MapIterator(out_df, block=True)
        for item in it:
            self.assertEqual(item[0][0], res)
            res += 1

        self.assertEqual(runner.status, RunnerStatus.FINISHED)

    def test_map_runner_with_error(self):
        input_df, _, runner = self._create_test_obj()
        runner.set_op(zero_drop.ZeroDrop())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        input_df.put({'num': 'error_data'})
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FAILED)

if __name__ == '__main__':
    unittest.main()
