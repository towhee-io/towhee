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
import logging
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

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
        hub_op_id:
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

    def __init__(self, op_name: str, hub_op_id: str, op_args: Dict[str, Any],
                 inputs: Dict[str, Any], task_idx: int):
        self._op_name = op_name
        self._hub_op_id = hub_op_id
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
    def hub_op_id(self) -> str:
        return self._hub_op_id

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

    @property
    def op_key(self) -> Tuple:
        """Calculates a unique key given the operator hub ID and the operator's
        initialization arguments. Operators with the same hub ID but different
        initialization arguments are considered separate.

        Returns:
            (`tuple`)
                Unique value corresponding to a combination of operator ininitialization
                arguments and its operator ID in the hub.
        """
        args_tup = tuple((key, self.op_args[key])
                         for key in sorted(self.op_args))
        return (self.hub_op_id, ) + args_tup

    def add_ready_handlers(self, handlers):
        """Adds a ready handler to this `Task` object.

        Args:
            handlers: (`typing.Callable`, or `list` of `typing.Callable`)
                Handlers that will be called once the task is ready to be executed.
        """
        if isinstance(handlers, list):
            self._on_ready_handlers += handlers
        else:
            self._on_ready_handlers.append(handlers)

    def add_start_handler(self, handler):
        """Adds a start handler to this `Task` object.

        Args:
            handler: (`typing.Callable`, or `list` of `typing.Callable`)
                Handler that will be called prior to execution.
        """
        if isinstance(handler, list):
            self._on_start_handlers += handler
        else:
            self._on_start_handlers.append(handler)

    def add_finish_handler(self, handler):
        """Adds a finish handler to this `Task` object.

        Args:
            handler: (`typing.Callable`, or `list` of `typing.Callable`)
                Handler that will be called when the `outputs` attribute is set.
        """
        if isinstance(handler, list):
            self._on_finish_handlers += handler
        else:
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
        try:
            self._outputs = op(**self._inputs)
        except Exception as e:  # pylint: disable=broad-except
            logging.error(e)
            self._outputs = None

        self._runtime = timeit.default_timer() - start
        self._execute_handlers(self._on_finish_handlers)
