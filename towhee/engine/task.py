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


import timeit
from typing import Callable

from towhee.operator.operator import OperatorBase


class Task:
    """Tasks represent containers for which input, output, and operator data is stored.
    Tasks themselves have no functioning logic, and merely serve as a container for the
    `TaskExecutor`.

    Args:
        op_name:
            Instead of each task executing its own operator, the Task maintains a
            operator name which can be used by executors to lookup the proper `Operator`
            to execute.
        inputs:
            A tuple of data, which serve as inputs to the operator call. It must be in
            the same order of parameters as operator.__call__().
        task_idx:
            A new task will be constructed for each operator call. The tasks
            are indexed individually for each operation performed, starting from 0.
    """

    def __init__(self, op_name: str, op_func: str, inputs: tuple, task_idx: int):
        self._op_name = op_name
        self._op_func = op_func
        self._inputs = inputs
        self._task_idx = task_idx

        self._outputs = None
        self._runtime = -1

        self._on_ready_handlers = []
        self._on_start_handlers = []
        self._on_finish_handlers = []

    @property
    def inputs(self) -> tuple:
        return _inputs

    @property
    def op_name(self) -> str:
        return _op_name

    @property
    def output(self) -> tuple:
        return self._outputs

    @property
    def runtime(self) -> float:
        return self._runtime

    def add_ready_handler(self, handler: Callable):
        """Adds a ready handler to this `Task` object.

        Args:
            handler: (`typing.Callable`)
                Ready handler that will be called once the task is ready to be executed.
        """
        self._on_ready_handlers.append(handler)

    def add_start_handler(self, handler: Callable):
        """Adds a start handler to this `Task` object.

        Args:
            handler: (`typing.Callable`)
                Handler that will be called prior to execution.
        """
        self._on_start_handlers.append(handler)

    def add_finish_handler(self, handler: Callable):
        """Adds a finish handler to this `Task` object.

        Args:
            handler: (`typing.Callable`)
                Handler that will be called when the `outputs` attribute is set.
        """
        self._on_finish_handlers.append(handler)

    def _execute_handlers(self, handlers: list[Callable]):
        """Execute handlers defined in `handlers`. These should be a list of callables
        which take this `Task` object as its input.

        Args:
            handlers: (`list[typing.Callable]`)
                A list of functions to be executed. Should be one of
                `_on_ready_handlers`, `_on_start_handlers`, and `_on_finish_handlers`.
        """
        for handler in handlers:
            handler(self)

    def execute(self, op: OperatorBase):
        """Given a corresponding `Operator` from the `TaskExecutor`, run the task.
        """
        self._execute_handlers(self._on_start_handlers)
        start = timeit.default_timer()

        # Run the operator. The graph provided to the engine should already have done
        # graph validity checks, so further input/output argument checks are
        # unnecessary.
        self._outputs = op(**self._inputs)

        self._runtime = timeit.default_timer() - start
        self._execute_handlers(self._on_finish_handlers)
