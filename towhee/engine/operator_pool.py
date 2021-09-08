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


from towhee.operator.operator import OperatorBase
from towhee.engine.operator_loader import OperatorLoader


class OperatorPool:
    """`OperatorPool` manages `Operator` creation, acquisition, release, and garbage
    collection. Each `TaskExecutor` has one `OperatorPool`.
    """

    def __init__(self):
        self._op_loader = OperatorLoader()
        self._all_ops = {}

    @property
    def available_ops(self):
        return self._all_ops.keys()

    def acquire_op(self, task: Task, args: dict = None) -> OperatorBase:
        """Given a `Task`, instruct the `OperatorPool` to reserve and return the
        specified operator for use in the executor.

        Args:
            name: (`str`)
                Unique operator name, as specified in the graph.
            args: (`dict`)
                The operator's initialization arguments, if any.

        Returns:
            (`towhee.OperatorBase`)
                The operator instance reserved for the caller.
        """
        if name not in self._all_ops:
            op = self._op_loader.load_operator(task.op_func, args)
        else:
            op = self._all_ops[task.op_name]

        # Let there be a record of the operator existing in the pool, but remove its
        # pointer until the operator is released by the `TaskExecutor`.
        self._all_ops[task.op_name] = None

    def release_op(self, op: Operator):
        """Releases the specified operator and all associated resources back to the
        `OperatorPool`.

        Args:
            op: (`towhee.OperatorBase`)
                `OperatorBase` instance to add back into the operator pool.
        """
        self._all_ops[op.name] = op
