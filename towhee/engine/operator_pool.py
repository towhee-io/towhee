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

from towhee.operator import Operator
from towhee.engine.operator_loader import OperatorLoader
from towhee.engine.task import Task


class OperatorPool:
    """`OperatorPool` manages `Operator` creation, acquisition, release, and garbage
    collection. Each `TaskExecutor` has one `OperatorPool`.
    """

    def __init__(self, cache_path: str = None):
        self._op_loader = OperatorLoader(cache_path)
        self._all_ops = {}

    @property
    def available_ops(self):
        return self._all_ops.keys()

    def is_op_available(self, task: Task) -> bool:
        """Determines whether an operator that can be used to fulfill the given Task is
        currently loaded.

        Args:
            task: (`towhee.Task`)
                Task

        Returns:
            (`bool`)
                Returns `True` if the specified input task can be run without having to
                load a new operator from cache.
        """
        return task.op_key in self._all_ops

    def acquire_op(self, task: Task) -> Operator:
        """Given a `Task`, instruct the `OperatorPool` to reserve and return the
        specified operator for use in the executor.

        Args:
            task: (`towhee.Task`)
                Task to acquire an operator for.

        Returns:
            (`towhee.operator.Operator`)
                The operator instance reserved for the caller.
        """

        # Load the operator if the computed key does not exist in the operator
        # dictionary.
        op_key = task.op_key
        if op_key not in self._all_ops:
            op = self._op_loader.load_operator(task.hub_op_id, task.op_args)
            op.key = op_key
        else:
            op = self._all_ops[op_key]

        # Let there be a record of the operator existing in the pool, but remove its
        # pointer until the operator is released by the `TaskExecutor`.
        self._all_ops[op_key] = None

        return op


    def acquire_op_v2(self, op_key: str, hub_op_id: str,
                      op_args: Dict[str, any]) -> Operator:
        """Given a `Task`, instruct the `OperatorPool` to reserve and return the
        specified operator for use in the executor.

        Args:
            task: (`towhee.Task`)
                Task to acquire an operator for.

        Returns:
            (`towhee.operator.Operator`)
                The operator instance reserved for the caller.
        """

        # Load the operator if the computed key does not exist in the operator
        # dictionary.
        if op_key not in self._all_ops:
            op = self._op_loader.load_operator(hub_op_id, op_args)
            op.key = op_key
        else:
            op = self._all_ops[op_key]

        # Let there be a record of the operator existing in the pool, but remove its
        # pointer until the operator is released by the `TaskExecutor`.
        self._all_ops[op_key] = None

        return op

    def release_op(self, op: Operator):
        """Releases the specified operator and all associated resources back to the
        `OperatorPool`.

        Args:
            op: (`towhee.Operator`)
                `Operator` instance to add back into the operator pool.
        """
        self._all_ops[op.key] = op
