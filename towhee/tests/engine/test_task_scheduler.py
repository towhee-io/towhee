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

from towhee.engine.task_executor import TaskExecutor
from towhee.engine.task_scheduler import FIFOTaskScheduler
from towhee.tests.emulated_pipeline import EmulatedPipeline


@unittest.skip('Scheduler changed')
class TestFIFOTaskScheduler(unittest.TestCase):
    """Basic test case for `FIFOTaskScheduler`.
    """

    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._task_execs = []
        for _ in range(4):
            self._task_execs.append(TaskExecutor('', cache_path=cache_path))
        self._pipeline = EmulatedPipeline()
        self._task_sched = FIFOTaskScheduler(self._task_execs)

    def tearDown(self):
        for task_exec in self._task_execs:
            task_exec.stop()

        # Ensure that exactly `n_runs` tasks have been completed after joining.
        self.assertEqual(self._runs_count, self._n_runs)

    def test_scheduler(self, n_runs=1024):

        # Add graph contexts to scheduler.
        for graph_ctx in self._pipeline.graph_contexts:
            self._task_sched.register(graph_ctx)

        # Add callback function upon completion.
        self._n_runs = n_runs
        self._runs_count = 0

        def _add_task_finish_callback(task):
            self._runs_count += 1  # No need for threading.Lock due to GIL.
            self.assertEqual(task.outputs.sum, task.inputs['num'])
        self._pipeline.add_task_finish_handler(_add_task_finish_callback)

        # Spin up executors.
        for task_exec in self._task_execs:
            task_exec.start()

        # Add callback function upon completion.
        for _ in range(n_runs):
            self._task_sched.schedule_step()


if __name__ == '__main__':
    unittest.main()
