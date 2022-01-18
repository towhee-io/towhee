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
from towhee.engine.operator_runner.flatmap_runner import FlatMapRunner
from tests.unittests.mock_operators.repeat_operator.repeat_operator import RepeatOperator

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


class TestFlatmapRunner(unittest.TestCase):
    """
    MapRunner test
    """
    def test_flatmap_runner(self):
        data_queue = Queue()
        writer = MockWriter()

        runner = FlatMapRunner('flattest', 0, 'repeat_operator', 'main', 'mock_operators', {'repeat': 3}, [MockReader(data_queue)], writer)

        runner.set_op(RepeatOperator(3))
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        data_queue.put({'num': 1})
        data_queue.put({'num': 2})
        data_queue.put({'num': 3})
        data_queue.put(None)
        t.join()
        self.assertEqual(runner.status, RunnerStatus.IDLE)

        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        data_queue.put({'num': 4})
        data_queue.put({'num': 5})
        data_queue.put(StopFrame())
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FINISHED)
        self.assertEqual(len(writer.res), 15)

    def test_flatmap_runner_with_error(self):
        data_queue = Queue()
        writer = MockWriter()
        runner = FlatMapRunner('test', 0, 'repeat_operator', 'main', 'mock_operators', {'repeat': 3}, [MockReader(data_queue)], writer)

        runner.set_op(RepeatOperator(3))
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        data_queue.put('error_data')
        runner.join()
        self.assertEqual(runner.status, RunnerStatus.FAILED)
