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


from typing import Callable, Dict, NamedTuple

from towhee.engine.task import Task


class EmulatedPipeline:
    """Dummy pipeline with only fields required by the FIFOTaskScheduler filled in.
    """

    def __init__(self):
        GraphContext = NamedTuple('GraphContext', [('operator_contexts', Dict)])
        OperatorContext = NamedTuple('OperatorContext', [('pop_ready_tasks', Callable), ('is_schedulable', Callable)])

        self._on_ready_handlers = []
        self._on_start_handlers = []
        self._on_finish_handlers = []
        self._task_idx = 0

        # Create dummy GraphContext with a single OperatorContext instance.
        self.graph_contexts = [GraphContext({'op': OperatorContext(self._pop_ready_tasks, self._is_schedulable)})]

    def _add_task_handlers(self, task: Task):
        """Helper function which adds handlers to the specified task.
        """
        for handler in self._on_ready_handlers:
            task.add_ready_handler(handler)
        for handler in self._on_start_handlers:
            task.add_start_handler(handler)
        for handler in self._on_finish_handlers:
            task.add_finish_handler(handler)

    def _pop_ready_tasks(self, n_tasks: int = 1):
        """Returns a list of of `add` tasks with continuously increasing `task_idx`.
        """
        hub_op_id = 'mock_operators/add_operator'
        args = {'factor': 0}
        tasks = []
        for n in range(n_tasks):
            num = self._task_idx + n
            task = Task('test', hub_op_id, args, {'num': num}, num)
            self._add_task_handlers(task)
            tasks.append(task)
        self._task_idx += n_tasks
        return tasks

    def _is_schedulable(self):
        return True

    def add_task_ready_handler(self, function: Callable):
        self._on_ready_handlers.append(function)

    def add_task_start_handler(self, function: Callable):
        self._on_start_handlers.append(function)

    def add_task_finish_handler(self, function: Callable):
        self._on_finish_handlers.append(function)
