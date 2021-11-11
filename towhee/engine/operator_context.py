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
from typing import List, Dict

from towhee.dag.operator_repr import OperatorRepr
from towhee.utils import HandlerMixin
from towhee.engine.task import Task
# from towhee.engine.graph_context import GraphContext
from towhee.dataframe import DataFrame
from towhee.engine._operator_io import create_reader, create_writer, create_multireader


class OperatorContext(HandlerMixin):
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
    """

    def __init__(
        self,
        op_repr: OperatorRepr,
        dataframes: Dict[str, DataFrame]
    ):
        self._repr = op_repr

        # todo: GuoRentong, issue #114
        inputs = list({dataframes[input['df']] for input in op_repr.inputs})
        iter_type = op_repr.iter_info['type']
        inputs_index = dict((item['name'], item['col'])
                            for item in op_repr.inputs)

        reader_inputs = {}
        for x in op_repr.inputs:
            if x['df'] in reader_inputs.keys():
                reader_inputs[x['df']]['cols'].append((x['name'], x['col']))
            else:
                reader_inputs[x['df']] = {}
                reader_inputs[x['df']]['df'] = dataframes[x['df']]
                reader_inputs[x['df']]['cols'] = [(x['name'], x['col'])]


        self.inputs = inputs
        if 'multi' in iter_type.lower():
            self._reader = create_multireader(reader_inputs, iter_type)
        else:
            self._reader = create_reader(inputs, iter_type, inputs_index)

        outputs = list({dataframes[output['df']]
                       for output in op_repr.outputs})
        self._writer = create_writer(iter_type, outputs)
        self.outputs = outputs

        self._lock = threading.Lock()
        self._finished = False
        self._num_finished_tasks = 0
        self._has_tasks = True
        self._task_idx = 0

        self.add_handler_methods(
            'op_start', 'op_finish', 'task_ready', 'task_start', 'task_finish')
        self.add_task_finish_handler(self._write_outputs)

    @property
    def name(self):
        return self._repr.name

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

            # None` implies that an input `DataFrame` has been sealed.
            if op_input_params is None:
                with self._lock:
                    self._has_tasks = False
                    if self._num_finished_tasks == self._task_idx:
                        self._finished = True
                        self._writer.close()

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
            self._writer.write(task.outputs)
            self._num_finished_tasks += 1

    def _next_task_inputs(self):
        """
        Manage the preparation works for an operator's inputs
        Returns one task's inputs on each call.
        Return: a list of inputs, list element can be scalar or Array.
        """
        raise NotImplementedError

    def _create_new_task(self, inputs: Dict[str, any]):
        t = Task(self.name, self._repr.function,
                 self._repr.init_args, inputs, self._task_idx)
        self._task_idx += 1
        t.add_task_finish_handler(self.task_finish_handlers)
        return t