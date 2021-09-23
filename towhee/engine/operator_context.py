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

from typing import List, Dict

# from collections import namedtuple
from towhee.engine.task import Task
# from towhee.engine.graph_context import GraphContext
from towhee.dataframe import DataFrame
from towhee.engine._operator_io import create_reader, create_writer


class OperatorContext:
    """
    The OperatorContext manages an operator's input data and output data at runtime,
    as well as the operators' dependency within a GraphContext.
    The abstraction of OperatorContext hides the complexity of Dataframe management,
    input iteration, and data dependency between Operators. It offers a Task-based
    scheduling context.
    """

    def __init__(self, op_info: Dict, inputs: List[DataFrame],
                 outputs: List[DataFrame]) -> None:
        """
        Args:
            op_info: op config, op init data, op input info
            inputs: a list of DataFrame.
            outputs: a list of DataFrame.
        """
        self._op_info = op_info
        self._reader = create_reader(
            inputs, op_info['iter_type'], op_info['inputs_index'])
        self._writer = create_writer(outputs)
        self._finished = False
        self._taskid = 0

    def pop_ready_tasks(self, n_tasks: int = 1) -> List:
        """
        Pop n ready Tasks if any. The number of returned Tasks may be less than n
        if there are not enough Tasks.

        Return: a list of ready Tasks.
        """
        ready_tasks = []
        task_num = n_tasks
        while task_num > 0:
            op_input_params = self._reader.read()
            if op_input_params:
                task = self._create_new_task(op_input_params)
                ready_tasks.append(task)
                task_num -= 1
                continue

            if op_input_params is None:
                self._finished = True
            break

        return ready_tasks

    def finished(self) -> bool:
        return self._finished

    @property
    def num_ready_tasks(self) -> int:
        """
        Get the number of ready Tasks.
        """
        return self._reader.size

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

    def write_outputs(self, task: Task):
        self._writer.write(task.outputs)

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

    def _create_new_task(self, inputs: Dict[str, any]):
        t = Task(self._op_info['name'], self._op_info['name'],
                 self._op_info['op_args'], inputs, self._taskid)
        self._taskid += 1
        t.add_finish_handler(self.write_outputs)
        return t

    def _notify_downstream_op_ctx(self):
        """
        When a Task at this Operator is finished, its outputs will be feeded to the
        downstream Operator, and followed by a notification.
        """
        raise NotImplementedError
