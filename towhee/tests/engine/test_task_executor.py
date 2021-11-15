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


from pathlib import Path
import unittest
# from shutil import rmtree

from towhee.engine.task_executor import TaskExecutor
from towhee.engine.task import Task
from towhee.hub.file_manager import FileManagerConfig, FileManager
from towhee.tests import CACHE_PATH

class TestTaskExecutor(unittest.TestCase):
    """Basic test case for `TaskExecutor`.
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
        fm = FileManager(fmc) # pylint: disable=unused-variable

    # @classmethod
    # def tearDownClass(cls):
    #     new_cache = (CACHE_PATH/'test_cache')
    #     rmtree(str(new_cache))

    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._task_exec = TaskExecutor('', cache_path=cache_path)
        self._task_exec.start()

    def tearDown(self):
        self._task_exec.stop()

    def test_add_task_execution(self):

        # Add callback function upon completion.
        def _add_task_finish_callback(task):
            self.assertEqual(task.outputs.sum, task.inputs['num'])

        # Create a couple of tasks to execute through the executor.
        tasks = []
        hub_op_id = 'local/add_operator'
        args = {'factor': 0}
        tasks.append(Task('test', hub_op_id, args, {'num': 0}, 0))
        tasks.append(Task('test', hub_op_id, args, {'num': 1}, 1))
        tasks.append(Task('test', hub_op_id, args, {'num': 10}, 10))

        # Add finish callbacks and submit the tasks to the executor.
        for task in tasks:
            task.add_task_finish_handler(_add_task_finish_callback)
            self._task_exec.push_task(task)

    def test_sub_task_execution(self):

        # Add callback function upon completion.
        def _add_task_finish_callback(task):
            diff = task.inputs['a'] - task.inputs['b']
            self.assertEqual(task.outputs.diff, diff)

        # Create a couple of tasks to execute through the executor.
        tasks = []
        hub_op_id = 'local/sub_operator'
        tasks.append(Task('test', hub_op_id, {}, {'a': 0, 'b': 0}, 0))
        tasks.append(Task('test', hub_op_id, {}, {'a': 10, 'b': 20}, 1))
        tasks.append(Task('test', hub_op_id, {}, {'a': 23, 'b': -1}, 24))

        # Add finish callbacks and submit the tasks to the executor.
        for task in tasks:
            task.add_task_finish_handler(_add_task_finish_callback)
            self._task_exec.push_task(task)


if __name__ == '__main__':
    unittest.main()
