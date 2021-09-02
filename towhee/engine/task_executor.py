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
from towhee.engine.operator_pool import OperatorPool
from towhee.operator.operator import Operator
from towhee.engine.task import Task
from towhee.engine.task_queue import TaskQueue


class TaskExecutor(threading.Thread):
    """
    A FIFO task executor. 
    Each device has one TaskExecutor.
    """
    def __init__(self, name: str):
        threading.Thread.__init__(self)
        self.name = name
        self._task_queue = TaskQueue()
        self._op_pool = OperatorPool()
        self._need_stop = False
    
    def acquire_op(self, name: str, init_args: dict) -> Operator:
        """
        Get an Operator from pool by name

        Args:
            name: the Operator's unique name in hub
            init_args: the Operator's initialization arguments
        """
        raise NotImplementedError
    
    def release_op(self, op: Operator):
        """
        Release an Operator to pool
        """
        raise NotImplementedError
    
    # def set_op_parallelism(self, name: str, parallel: int = 1):
    #     """
    #     Set how many Operators with same name can be called concurrently.
    #     """
    #     raise NotImplementedError
    
    def push_task(self, task: Task):
        """
        Push a task to task queue
        """
        # todo self._task_queue.push(task)
        raise NotImplementedError

    def run(self):
        while not self._need_stop:
            task = self._task_queue.pop()
            if task != None:
                task.run()
        raise NotImplementedError

    def _get_op(self, task: Task):
        """
        Get the operator on which the Task will perform.
        """
        raise NotImplementedError