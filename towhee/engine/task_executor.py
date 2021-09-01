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
<<<<<<< HEAD
from towhee.engine.operator_pool import OperatorPool
=======
>>>>>>> 9ada236dd1ea0de151ebd569a018da7ffce5cac7
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
<<<<<<< HEAD
        self._op_pool = OperatorPool()
        self._need_stop = False
    
    def acquire_op(self, name: str) -> Operator:
        """
        Get an Operator from pool by name

        Args:
            name: the Operator's unique name in hub
        """
        raise NotImplementedError
    
    def release_op(self, op: Operator):
        """
        Release an Operator to pool
        """
        raise NotImplementedError
    
    def set_op_parallelism(self, name: str, parallel: int = 1):
        """
        Set how many Operators with same name can be called concurrently.
        """
=======
        self._ops = []
        self._need_stop = False
    
    def add_op(self, op: Operator):
        """
        Add an Operator to the TaskExecutor and load the Operator to device memory.
        """
        self._ops.append(op)
>>>>>>> 9ada236dd1ea0de151ebd569a018da7ffce5cac7
        raise NotImplementedError
    
    def push_task(self, task: Task):
        """
        Push a task to task queue
        """
<<<<<<< HEAD
        # todo self._task_queue.push(task)
=======
        op = self._get_op(task)
        if op != None:
            task.op = op
            self._task_queue.push(task)
>>>>>>> 9ada236dd1ea0de151ebd569a018da7ffce5cac7
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