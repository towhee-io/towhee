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
import time

from towhee.engine.operator_runner.map_runner import MapRunner
from towhee.engine.operator_runner.runner_base import RunnerStatus
from towhee.engine.thread_pool_task_executor import ThreadPoolTaskExecutor
from towhee.engine.operator_io import create_reader, create_writer
from towhee.dataframe import DataFrame
from towhee.dataframe.iterators import MapIterator
from towhee.hub.file_manager import FileManagerConfig, FileManager
from tests.unittests import CACHE_PATH


class TestThreadPoolTaskExecutor(unittest.TestCase):
    """
    Thread pool task executor test
    """
    @classmethod
    def setUpClass(cls):
        new_cache = (CACHE_PATH / 'test_cache')
        pipeline_cache = (CACHE_PATH / 'test_util')
        operator_cache = (CACHE_PATH / 'mock_operators')
        fmc = FileManagerConfig()
        fmc.update_default_cache(new_cache)
        pipelines = list(pipeline_cache.rglob('*.yaml'))
        operators = [f for f in operator_cache.iterdir() if f.is_dir()]
        fmc.cache_local_pipeline(pipelines)
        fmc.cache_local_operator(operators)
        FileManager(fmc)

    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._task_exec = ThreadPoolTaskExecutor('tread_pool_task_executor_test', cache_path)
        self._task_exec.start()

    def tearDown(self):
        if self._task_exec.is_alive():
            self._task_exec.stop()
            self._task_exec.join()

    def _create_test_obj(self):
        input_df = DataFrame('input', [('num', 'int')])
        out_df = DataFrame('output', [('sum', 'int')])
        reader = create_reader(input_df, 'map', {'num': 0})
        writer = create_writer('map', [out_df])
        hub_op_id = 'local/add_operator'
        runner = MapRunner('test', 0, 'add_operator', 'main', hub_op_id, {'factor': 1}, [reader], writer)
        return input_df, out_df, runner

    def test_pool_with_map_runner(self):
        input_df, out_df, runner = self._create_test_obj()
        self._task_exec.push_task(runner)

        input_df.put({'num': 1})
        input_df.put({'num': 2})
        input_df.put({'num': 3})
        input_df.seal()

        time.sleep(0.1)
        runner.set_stop()
        time.sleep(0.1)
        self._task_exec.stop()
        out_df.seal()

        res = 2
        it = MapIterator(out_df)
        for item in it:
            self.assertEqual(item[0][0], res)
            res += 1

        self.assertEqual(runner.status, RunnerStatus.FINISHED)

    def test_pool_with_map_runner_error(self):
        input_df, out_df, runner = self._create_test_obj()
        self._task_exec.push_task(runner)
        input_df.put({'num': 'error'})
        input_df.seal()
        time.sleep(0.1)
        runner.set_stop()
        time.sleep(0.1)
        self._task_exec.stop()
        self.assertEqual(out_df.size, 0)
        self.assertEqual(runner.status, RunnerStatus.FAILED)

if __name__ == '__main__':
    unittest.main()
