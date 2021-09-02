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


from collections import namedtuple
from towhee.engine.task import Task
from towhee.engine.variable import Variable


class OperatorContext:
    """
    The OperatorContext manages an operator's input data and output data at runtime, 
    as well as the operators' dependency within a GraphContext.
    The abstraction of OperatorContext hides the complexity of Dataframe management, 
    input iteration, and data dependency between Operators. It offers a Task-based 
    scheduling context.
    """

    def __init__(self, graph_ctx, inputs: list, outputs: list):
        """
        Args:
            inputs: a list of Variable.
            outputs: a list of Variable.
        """
        self._inputs = inputs
        self._outputs = outputs
        self.on_task_finish_handlers = []
        raise NotImplementedError
   
    def pop_ready_tasks(self, n: int = 1) -> list:
        """
        Pop n ready Tasks if any. The number of returned Tasks may be less than n
        if there are not enough Tasks.

        Return: a list of ready Tasks.
        """

        # create a new task
        task.on_finish_handlers.append(self.on_task_finish_handlers)
        raise NotImplementedError
 
    @property
    def num_ready_tasks(self) -> int:
        """
        Get the number of ready Tasks.
        """
        raise NotImplementedError
        # consider the thread-safe read write. This OperatorContext should be
        # self._ready_tasks' only monifier.
    
    @property
    def num_finished_tasks(self) -> int:
        """
        Get the number of finished tasks.
        """
        raise NotImplementedError
        # consider the thread-safe read write. This OperatorContext should be
        # self._finished_tasks' only monifier.

    def _on_task_start_handler(self, task: Task):
        """
        The handler for the event of task start.
        """
        raise NotImplementedError

    def _on_task_finish_handler(self, task: Task):
        """
        The handler for the event of task finish.
        Feed downstream operator's inputs with this task's results.
        """
        raise NotImplementedError

    def _next_task_inputs(self):
        """
        Manage the preparation works for an operator's inputs
        Returns one task's inputs on each call.

        Return: a list of inputs, list element can be scalar or Array.
        """
        raise NotImplementedError
    
    def _create_new_task(self, inputs: list):
        #t = Task()
        # todo: setup t
        #t.on_start.append(self._on_task_start)
        raise NotImplementedError
    
    def _notify_downstream_op_ctx(self):
        """
        When a Task at this Operator is finished, its outputs will be feeded to the
        downstream Operator, and followed by a notification.
        """
        raise NotImplementedError
