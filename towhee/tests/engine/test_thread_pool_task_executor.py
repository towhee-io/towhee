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
from pathlib import Path
from typing import Dict
from queue import Queue
import time

from towhee.engine.operator_runner.map_runner import MapRunner
from towhee.engine.operator_runner.runner_base import RunnerStatus
from towhee.engine.thread_pool_task_executor import ThreadPoolTaskExecutor
from towhee.engine.operator_io._mock_reader import MockReader
from towhee.hub.file_manager import FileManagerConfig, FileManager
from towhee.tests import CACHE_PATH


DATA_QUEUE = Queue()


class MockWriter:
    def __init__(self):
        self.res = []

    def write(self, data: Dict):
        self.res.append(data)


class TestThreadPoolTaskExecutor(unittest.TestCase):
    """
    Thread pool task executor test
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

    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._task_exec = ThreadPoolTaskExecutor('tread_pool_task_executor_test',
                                                 cache_path)
        self._task_exec.start()

    def tearDown(self):
        if self._task_exec.is_alive():
            self._task_exec.stop()
            self._task_exec.join()

    def test_pool_with_map_runner(self):
        data_queue = Queue()
        writer = MockWriter()
        hub_op_id = 'local/add_operator'
        runner = MapRunner('test', 0, 'add_operator',
                           hub_op_id, {'factor': 1},
                           [MockReader(data_queue)], writer)
        self._task_exec.push_task(runner)

        data_queue.put({'num': 1})
        data_queue.put({'num': 2})
        data_queue.put({'num': 3})

        time.sleep(0.1)
        runner.set_stop()
        time.sleep(0.1)
        self._task_exec.stop()

        res = 2
        for item in writer.res:
            self.assertEqual(item[0], res)
            res += 1

        self.assertEqual(runner.status, RunnerStatus.FINISHED)

    def test_pool_with_map_runner_error(self):
        data_queue = Queue()
        writer = MockWriter()
        hub_op_id = 'local/add_operator'
        runner = MapRunner('test', 0, 'add_operator',
                           hub_op_id, {'factor': 1},
                           [MockReader(data_queue)], writer)
        self._task_exec.push_task(runner)

        data_queue.put('error')
        time.sleep(0.1)
        runner.set_stop()
        time.sleep(0.1)
        self._task_exec.stop()
        self.assertEqual(len(writer.res), 0)
        self.assertEqual(runner.status, RunnerStatus.FAILED)
