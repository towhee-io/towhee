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
import threading

from towhee.operator import Operator, SharedType
from towhee.engine.operator_loader import OperatorLoader


class _OperatorStorage:
    """
    Impl operator get and put by different shared_type.
    """
    def __init__(self):
        self._shared_type = None
        self._ops = []

    def op_available(self) -> bool:
        return self._shared_type is not None and len(self._ops) > 0

    def get(self):
        assert self._shared_type is not None and len(self._ops) != 0
        op = self._ops[-1]
        if self._shared_type != SharedType.Shareable:
            self._ops.pop()
        return op

    def put(self, op: Operator, force_put: bool = False):
        if self._shared_type is None:
            self._shared_type = op.shared_type

        if force_put or self._shared_type == SharedType.NotShareable:
            self._ops.append(op)

    def __len__(self):
        return len(self._ops)


class OperatorPool:
    """`OperatorPool` manages `Operator` creation, acquisition, release, and garbage
    collection. Each `TaskExecutor` has one `OperatorPool`.
    """
    def __init__(self, cache_path: str = None):
        self._op_loader = OperatorLoader(cache_path)
        self._all_ops = {}
        self._lock = threading.Lock()

    @staticmethod
    def _operator_id(hub_op_id: str, op_args: Dict[str, any], tag: str):
        if op_args:
            args_tup = tuple(sorted(op_args.items()))
        else:
            args_tup = ()
        return (hub_op_id, tag) + args_tup

    def __len__(self):
        num = 0
        for _, op_storage in self._all_ops.items():
            num += len(op_storage)
        return num

    def clear(self):
        self._all_ops = {}

    def acquire_op(self, hub_op_id: str, op_args: Dict[str, any], tag: str) -> Operator:
        """
        Instruct the `OperatorPool` to reserve and return the
        specified operator for use in the executor.

        Args:
            hub_op_id: (`str`)
            op_args: (`Dict[str, any]`)
                operator init parameters

        Returns:
            (`towhee.operator.Operator`)
                The operator instance reserved for the caller.
        """

        # Load the operator if the computed key does not exist in the operator
        # dictionary.
        op_key = OperatorPool._operator_id(hub_op_id, op_args, tag)
        with self._lock:
            storage = self._all_ops.get(op_key, None)
            if storage is None:
                storage = _OperatorStorage()
                self._all_ops[op_key] = storage

            if not storage.op_available():
                op = self._op_loader.load_operator(hub_op_id, op_args, tag)
                op.key = op_key
                storage.put(op, True)
            return storage.get()

    def release_op(self, op: Operator):
        """Releases the specified operator and all associated resources back to the
        `OperatorPool`.

        Args:
            op: (`towhee.Operator`)
                `Operator` instance to add back into the operator pool.
        """
        with self._lock:
            storage = self._all_ops[op.key]
            storage.put(op)
