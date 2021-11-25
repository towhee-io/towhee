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

from towhee.engine.operator_runner.runner_base import RunnerStatus
from towhee.engine.operator_runner.generator_runner import GeneratorRunner
from towhee.tests.mock_operators.generator_operator import generator_operator


DATA_QUEUE = Queue()


class StopFrame:
    pass


class MockReader:
    def __init__(self, queue: Queue):
        self._queue = queue

    def read(self):
        data = self._queue.get()
        if not isinstance(data, StopFrame):
            return data
        else:
            raise StopIteration()


class MockWriter:
    def __init__(self):
        self.res = []

    def write(self, data: Dict):
        self.res.append(data)


def run(runner):
    runner.process()


class TestGeneratorRunner(unittest.TestCase):
    """
    GeneratorRunner test
    """

    def test_map_runner(self):
        data_queue = Queue()
        writer = MockWriter()
        runner = GeneratorRunner('test', 0, 'generator_operator',
                                 'mock_operators', {},
                                 MockReader(data_queue), writer)
        runner.set_op(generator_operator.GeneratoOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        data_queue.put({'num': 10})
        data_queue.put(None)
        t.join()
        res = 0
        for item in writer.res:
            self.assertEqual(item[0], res)
            res += 1
        self.assertEqual(len(writer.res), 10)
        self.assertEqual(runner.status, RunnerStatus.IDLE)

        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        data_queue.put({'num': 4})
        data_queue.put({'num': 5})
        data_queue.put(StopFrame())
        runner.join()
        self.assertEqual(len(writer.res), 19)
        self.assertEqual(runner.status, RunnerStatus.FINISHED)

    def test_generator_runner_with_error(self):
        data_queue = Queue()
        writer = MockWriter()
        runner = GeneratorRunner('test', 0, 'generator_operator',
                                 'mock_operators', {},
                                 MockReader(data_queue), writer)
        runner.set_op(generator_operator.GeneratoOperator())
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        data_queue.put('error_data')
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FAILED)
