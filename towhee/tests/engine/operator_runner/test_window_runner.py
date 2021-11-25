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
from typing import Dict
import threading
from queue import Queue

from towhee.dataframe import DataFrame
from towhee.engine.operator_io.reader import BatchFrameReader
from towhee.engine.operator_runner.runner_base import RunnerStatus
from towhee.engine.operator_runner.window_runner import WindowRunner
from towhee.tests.mock_operators.sum_operator.sum_operator import SumOperator


DATA_QUEUE = Queue()


class MockWriter:
    def __init__(self):
        self.res = []

    def write(self, data: Dict):
        self.res.append(data)


def run(runner):
    runner.process()


class TestMapRunner(unittest.TestCase):
    """
    MapRunner test
    """

    def test_flatmap_runner(self):
        writer = MockWriter()

        df_in = DataFrame('op_test_in', {'num': {'type': 'int', 'index': 0}})

        # We
        runner = WindowRunner('window_test', 0, 'sum_operator',
                              'mock_operators', {},
                              BatchFrameReader(df_in, {'num': 0}, 5, 3),
                              writer)

        runner.set_op(SumOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        for _ in range(100):
            df_in.put_dict({'num': 1})
        df_in.seal()
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FINISHED)
        self.assertEqual(len(writer.res), 68)
        count = 0
        for item in writer.res:
            if count < 64:
                self.assertEqual(item.sum, 5)
            elif count < 66:
                self.assertEqual(item.sum, 4)
            else:
                self.assertEqual(item.sum, 1)
            count += 1

    def test_window_runner_with_error(self):
        writer = MockWriter()

        df_in = DataFrame('op_test_in', {'num': {'type': 'int', 'index': 0}})

        # We
        runner = WindowRunner('window_test', 0, 'sum_operator',
                              'mock_operators', {},
                              BatchFrameReader(df_in, {'num': 0}, 5, 3),
                              writer)

        runner.set_op(SumOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        df_in.put(('error_data',))
        df_in.seal()
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FAILED)
