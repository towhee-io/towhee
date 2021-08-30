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

    def __init__(self, subjob, task_idx, op_idx, inputs, profiling_on = False):
        """
        Args:
            subjob: the subjob this task belongs to.
            task_idx: a new task will be constructed for each operator call. The tasks
                are indexed individually on each opeartor, starting from 0.
            op_idx: on which operator the task will perform.
            inputs: the inputs of the operator call. It is a list of arguments, which
                has the same argument order of Operator.__call__.
            profiling_on: open the task profiling or not. If open, the time consumption
                and the resource consumption will be monitored.
        """
        self.task_idx = task_idx
        self.op = None
        self.inputs = inputs

        self._subjob = subjob
        self._on_start_handler = None
        self._on_finish_handler = None

        if profiling_on:
            self.time_cost = 0
        raise NotImplementedError
    
    def on_start(self, handler: function):
        """
        Set a custom handler that called before the execution of the task.
        """
        self._on_start_handler = handler
        raise NotImplementedError

    def on_finish(self, handler: function):
        """
        Set a custom handler that called after the execution of the task.
        """
        self._on_finish_handler = handler
        raise NotImplementedError

    def run(self):
        """
        Run the task
        """
        if self._on_start_handler != None:
            self._on_start_handler(self)

        self.outputs = self.op(*self.inputs)        

        if self._on_finish_handler != None:
            self._on_finish_handler(self)

        raise NotImplementedError

