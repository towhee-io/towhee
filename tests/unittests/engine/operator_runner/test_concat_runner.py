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
from towhee.engine.operator_runner.concat_runner import ConcatRunner
from towhee.operator.concat_operator import ConcatOperator

DATA_QUEUE = Queue()


class StopFrame:
    pass


class MockReader:
    """
    Mock reader
    """
    def __init__(self, queue: Queue):
        self._queue = queue

    def read(self):
        data = self._queue.get()
        if not isinstance(data, StopFrame):
            return data
        else:
            self._queue.put(StopFrame())
            raise StopIteration()


class MockWriter:
    def __init__(self):
        self.res = []

    def write(self, data: Dict):
        self.res.append(data)


def run(runner):
    runner.process()


class TestConcatRunner(unittest.TestCase):
    """
    Concat runner test
    """
    def test_concat_runner(self):
        data_queue_1 = Queue()
        data_queue_2 = Queue()
        data_queue_3 = Queue()
        writer = MockWriter()
        runner = ConcatRunner(
            'test',
            0,
            'add_operator',
            'main',
            'mock_operators', {'num': 1}, [MockReader(data_queue_1), MockReader(data_queue_2), MockReader(data_queue_3)],
            writer
        )
        runner.set_op(ConcatOperator('row'))
        t = threading.Thread(target=run, args=(runner, ))
        t.start()
        self.assertEqual(runner.status, RunnerStatus.RUNNING)
        res = []
        for i in range(3):
            data_queue_1.put({'num1': i})
            data_queue_2.put({'num2': i})
            data_queue_3.put({'num3': i})
            res.append({'num1': i, 'num2': i, 'num3': i})

        data_queue_2.put(StopFrame())
        for i in range(3):
            data_queue_1.put({'num1': i + 3})
            data_queue_3.put({'num3': i + 3})
            res.append({'num1': i + 3, 'num3': i + 3})
        data_queue_1.put(StopFrame())
        data_queue_3.put(StopFrame())

        runner.join()
        for i in range(len(writer.res)):
            self.assertEqual(writer.res[i], res[i])
            self.assertEqual(runner.status, RunnerStatus.FINISHED)

if __name__ == '__main__':
    unittest.main()
