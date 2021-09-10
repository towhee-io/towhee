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
from typing import Any, Callable, Dict, List, NamedTuple

from towhee.operator import Operator


class Task:
    """Tasks represent containers for which input, output, and operator data is stored.
    Tasks themselves have no functioning logic, and merely serve as a container for the
    `TaskExecutor`.

    Args:
        op_name:
            Instead of each task executing its own operator, the Task maintains a
            operator name which can be used by executors to lookup the proper `Operator`
            to execute.
        op_tag:
            Each operator in the operator hub is tagged with a string that describes its
            functionality.
        op_args:
            Input initialization arguments to the operator. Operators with the same tag
            but with different initialization arguments are considered separate
            operators within the same graph.
        inputs:
            A dictionary of keyward arguments which serve as inputs to the operator
            call.
        task_idx:
            A new task will be constructed for each operator call. The tasks
            are indexed individually for each operation performed, starting from 0.
    """

    def __init__(self, op_name: str, op_tag: str, op_args: Dict[str, Any],
                 inputs: Dict[str, Any], task_idx: int):
        self._op_name = op_name
        self._op_tag = op_tag
        self._op_args = op_args
        self._inputs = inputs
        self._task_idx = task_idx

        self._outputs = None
        self._runtime = -1

        self._on_ready_handlers = []
        self._on_start_handlers = []
        self._on_finish_handlers = []

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def op_tag(self) -> str:
        return self._op_tag

    @property
    def op_args(self) -> Dict[str, Any]:
        return self._op_args

    @property
    def inputs(self) -> Dict[str, Any]:
        return self._inputs

    @property
    def outputs(self) -> NamedTuple:
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

    def _execute_handlers(self, handlers: List[Callable]):
        """Execute handlers defined in `handlers`. These should be a list of callables
        which take this `Task` object as its input.

        Args:
            handlers: (`list[typing.Callable]`)
                A list of functions to be executed. Should be one of
                `_on_ready_handlers`, `_on_start_handlers`, and `_on_finish_handlers`.
        """
        for handler in handlers:
            handler(self)

    def execute(self, op: Operator):
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
