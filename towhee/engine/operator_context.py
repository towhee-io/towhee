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

from typing import Dict
from enum import Enum, auto

from towhee.dag.operator_repr import OperatorRepr
from towhee.engine.operator_runner.runner_base import RunnerStatus
from towhee.dataframe import DataFrame
from towhee.engine.operator_io import create_reader, create_writer
from towhee.engine.operator_runner import create_runner
from towhee.engine.thread_pool_task_executor import ThreadPoolTaskExecutor


class OpStatus(Enum):
    NOT_RUNNING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()
    STOPPED = auto()


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
    """
    def __init__(self, op_repr: OperatorRepr, dataframes: Dict[str, DataFrame]):
        self._repr = op_repr
        iter_type = op_repr.iter_info['type']
        reader_inputs = {}
        input_order = []
        for x in op_repr.inputs:
            # If a dataframe is already added, append the new columns.
            if x['df'] in reader_inputs.keys():
                reader_inputs[x['df']]['cols'].append((x['name'], x['col']))
                input_order.append(x['name'])
            else:
                reader_inputs[x['df']] = {}
                reader_inputs[x['df']]['df'] = dataframes[x['df']]
                reader_inputs[x['df']]['cols'] = [(x['name'], x['col'])]
                input_order.append(x['name'])

        self._reader = create_reader(reader_inputs, input_order, iter_type)

        outputs = list({dataframes[output['df']] for output in op_repr.outputs})

        self._writer = create_writer(iter_type, outputs)
        self._op_runners = []

        self._op_status = OpStatus.NOT_RUNNING
        self._err_msg = None

    @property
    def name(self):
        return self._repr.name

    @property
    def err_msg(self):
        return self._err_msg

    @property
    def status(self):
        """
        Calc op-ctx status by checking all runners of this op-ctx
        """
        if self._op_status in [OpStatus.FINISHED, OpStatus.FAILED]:
            return self._op_status

        if len(self._op_runners) == 0:
            return self._op_status

        finished_count = 0
        for runner in self._op_runners:
            if runner.status == RunnerStatus.FAILED:
                self._op_status = OpStatus.FAILED
                self._err_msg = runner.msg
            else:
                if runner.status == RunnerStatus.FINISHED:
                    finished_count += 1
        if finished_count == len(self._op_runners):
            self._op_status = OpStatus.FINISHED
        return self._op_status

    def start(self, executor: ThreadPoolTaskExecutor, count: int = 1) -> None:
        if self._op_status != OpStatus.NOT_RUNNING:
            raise RuntimeError('OperatorContext can only be started once')

        self._op_status = OpStatus.RUNNING

        for i in range(count):
            self._op_runners.append(
                create_runner(
                    self._repr.iter_info['type'],
                    self._repr.name,
                    i,
                    self._repr.name,
                    self._repr.function,
                    self._repr.init_args,
                    self._reader,
                    self._writer
                )
            )
        for runner in self._op_runners:
            executor.push_task(runner)

    def stop(self):
        if self.status != OpStatus.RUNNING:
            raise RuntimeError('Op ctx is already stopped.')

        for runner in self._op_runners:
            runner.set_stop()

    def join(self):
        # Waits all runner finished.
        for runner in self._op_runners:
            runner.join()
        self._writer.close()
