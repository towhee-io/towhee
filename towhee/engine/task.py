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


from towhee.operator.operator import Operator


class Task:
    """
    A task is a wrapper of an Operator call. 
    """

    def __init__(self, graph_ctx, task_idx, op, inputs, profiling_on = False):
        """
        Args:
            graph_ctx: the GraphContext containing this Task
            task_idx: a new task will be constructed for each operator call. The tasks
                are indexed individually on each opeartor, starting from 0.
            op: on which operator the task will perform.
            inputs: the inputs of the operator call. It is a list of arguments, which
                has the same argument order of Operator.__call__.
            profiling_on: open the task profiling or not. If open, the time consumption
                and the resource consumption will be monitored.
        """
        self.task_idx = task_idx
        self.op = None
        self.inputs = inputs

        self._graph_ctx = graph_ctx
        self.on_ready_handlers = []
        self.on_start_handlers = []
        self.on_finish_handlers = []

        if profiling_on:
            self.time_cost = 0
        raise NotImplementedError
    
    def _on_ready(self):
        """
        Callback when a task is ready.
        """
        for handler in self.on_ready_handlers:
            handler(self)

        raise NotImplementedError
   
    def _on_start(self, handler: function):
        """
        Callback before the execution of the task.
        """
        self._on_start_handler = handler
        raise NotImplementedError

    def _on_finish(self, handler: function):
        """
        Callback after the execution of the task.
        """
        self._on_finish_handler = handler
        raise NotImplementedError

    def run(self):
        """
        Run the task
        """
        self._on_start()

        self.outputs = self.op(*self.inputs)        

        self._on_finsh()

        raise NotImplementedError

