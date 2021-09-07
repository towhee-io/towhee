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


import threading

from towhee.operator.operator import Operator
from towhee.engine.operator_pool import OperatorPool
from towhee.engine.task import Task
from towhee.engine.task_queue import TaskQueue


class TaskExecutor(threading.Thread):
    """
    A FIFO task executor.
    Each device has one TaskExecutor.
    """

    def __init__(self, name: str):
        super().__init__(self)
        self._name = name
        self._task_queue = TaskQueue()
        self._op_pool = OperatorPool()
        self._do_run = False

    @property
    def name(self):
        return self._name

    # def set_op_parallelism(self, name: str, parallel: int = 1):
    #     """
    #     Set how many Operators with same name can be called concurrently.
    #     """
    #     raise NotImplementedError

    def push_task(self, task: Task) -> bool:
        """Push a task to the end of the task queue.

        Args:
            task:
                Pre-initialized `Task` object to push onto the queue.
        """
        return self._task_queue.push(task)

    def run(self):
        """Runs the execution loop.
        """
        while not self._do_run:

            # If there are no tasks in the queue, this might return `None` after
            # blocking for a while.
            task = self._task_queue.pop()

            # Execute tasks in first-in, first-out fashion. The `Scheduler` is
            # ultimately the one responsible for determining which tasks get executed
            # before others.
            if task:
                op = self._op_pool.acquire(task.op_name)
                task.execute(op)
                self._op_pool.release(op)
