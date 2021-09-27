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
import threading

from towhee.dag.operator_repr import OperatorRepr

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

    Args:
        op_repr: (OperatorRepr)
            The operator representation
        dataframes: (`dict` of `DataFrame`)
            All the `DataFrames` in `GraphContext`
        is_schedulable: (`bool`)
            Whether the `OperatorContext` is schedulable.
            There are special `OperatorContext`s that are not schedulable, such as
            `_start_op`, `_end_op`.

    """

    def __init__(
        self,
        op_repr: OperatorRepr,
        dataframes: Dict[str, DataFrame],
        is_schedulable: bool = True
    ):
        self._repr = op_repr
        self._is_schedulable = is_schedulable

        # todo: GuoRentong, issue #114
        inputs = list({dataframes[input['df']] for input in op_repr.inputs})
        input_iter_type = op_repr.iter_info['type']
        inputs_index = dict((item['name'], item['col'])
                            for item in op_repr.inputs)
        self.inputs = inputs
        self._reader = create_reader(inputs, input_iter_type, inputs_index)

        outputs = list({dataframes[output['df']]
                       for output in op_repr.outputs})
        self._writer = create_writer(outputs)
        self.outputs = outputs

        self._finished = False
        self._has_tasks = True
        self._taskid = 0
        self._finished_task_count = 0
        self._lock = threading.Lock()

        self.on_start_handlers = []
        self.on_finish_handlers = []

        self.on_task_ready_handlers = []
        self.on_task_start_handlers = []
        self.on_task_finish_handlers = [self._write_outputs]

    @property
    def name(self):
        return self._repr.name

    @property
    def is_schedulable(self) -> bool:
        return self._is_schedulable

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
                self._has_tasks = False

            if not self._has_tasks and self._taskid == self._finished_task_count:
                self._writer.close()
                self._finished = True
            break
        return ready_tasks

    @property
    def is_finished(self) -> bool:
        # todo: GuoRentong. see issue #124
        return self._finished

    @property
    def has_tasks(self) -> bool:
        # todo: GuoRentong. see issue #124
        return self._has_tasks

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

    def _write_outputs(self, task: Task):
        with self._lock:
            self._finished_task_count += 1
            self._writer.write(task.outputs)

    def _next_task_inputs(self):
        """
        Manage the preparation works for an operator's inputs
        Returns one task's inputs on each call.

        Return: a list of inputs, list element can be scalar or Array.
        """
        raise NotImplementedError

    def _create_new_task(self, inputs: Dict[str, any]):
        with self._lock:
            t = Task(self.name, self._repr.function,
                     self._repr.init_args, inputs, self._taskid)
            self._taskid += 1
            t.add_finish_handler(self.on_task_finish_handlers)
            return t
