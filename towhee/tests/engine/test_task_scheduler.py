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
from typing import Callable, List, NamedTuple
import unittest

from towhee.engine.task import Task
from towhee.engine.task_executor import TaskExecutor
from towhee.engine.task_scheduler import FIFOTaskScheduler


class _MockPipeline:
    """Dummy pipeline with only fields required by the FIFOTaskScheduler filled in.
    """

    def __init__(self):
        GraphContext = NamedTuple('GraphContext', [('op_ctxs', List)])
        OperatorContext = NamedTuple('OperatorContext', [('pop_ready_tasks', Callable)])

        # Initialize task list.
        self._tasks = []
        hub_op_id = 'mock_operators/add_operator'
        args = {'factor': 0}
        for n in range(10):
            self._tasks.append(Task('test', hub_op_id, args, {'num': n}, n))

        # Create dummy GraphContext with a single OperatorContext instance.
        self.graph_ctx = GraphContext([OperatorContext(self._pop_ready_task)])

    def _pop_ready_task(self):
        return self._tasks.pop() if self._tasks else None


class TestFIFOTaskScheduler(unittest.TestCase):
    """Basic test case for `FIFOTaskScheduler`.
    """

    def setUp(self):
        cache_path = Path(__file__).parent.parent.resolve()
        self._task_exec = TaskExecutor('', cache_path=cache_path)
        self._task_sched = FIFOTaskScheduler([self._task_exec])
        self._pipeline = _MockPipeline()

    def test_scheduler(self):
        self._task_sched.add_pipeline(self._pipeline)
        self._task_sched.schedule_step()


if __name__ == '__main__':
    unittest.main()
