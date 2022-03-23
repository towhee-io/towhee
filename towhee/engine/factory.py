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
import os
import threading
from typing import Tuple

from towhee.engine.operator_loader import OperatorLoader
from towhee.hparam import param_scope


def op(operator_src: str, tag: str = 'main', **kwargs):
    """
    Entry method which takes either operator tasks or paths to python files or class in notebook.
    An `Operator` object is created with the init args(kwargs).
    Args:
        operator_src (`str`):
            operator name or python file location or class in notebook.
        tag (`str`):
            Which tag to use for operators on hub, defaults to `main`.
    Returns
        (`typing.Any`)
            The `Operator` output.
    """
    if isinstance(operator_src, type):
        class_op = type('operator', (operator_src, ), kwargs)
        return class_op.__new__(class_op, **kwargs)

    loader = OperatorLoader()
    if os.path.isfile(operator_src):
        op_obj = loader.load_operator_from_path(operator_src, kwargs)
    else:
        op_obj = loader.load_operator(operator_src, kwargs, tag)

    return op_obj


class _OperatorLazyWrapper:
    """
    operator wrapper for lazy initialization.
    """

    def __init__(self,
                 _name: str,
                 index: Tuple[str],
                 tag: str = 'main',
                 **kws) -> None:
        self._name = _name.replace('.', '/').replace('_', '-')
        self._index = index
        self._tag = tag
        self._kws = kws
        self._op = None
        self._lock = threading.Lock()

    def __call__(self, *arg, **kws):
        with self._lock:
            if self._op is None:
                self._op = op(self._name, self._tag, **self._kws)

        if bool(self._index):
            # Multi inputs.
            if isinstance(self._index[0], tuple):
                args = []
                for i in self._index[0]:
                    args.append(getattr(arg[0], i))
                res = self._op(*args, **kws)
            # Single input.
            else:
                args = getattr(arg[0], self._index[0])
                res = self._op(args, **kws)

            # Multi outputs.
            if isinstance(res, tuple):
                if not isinstance(self._index[1],
                                  tuple) or len(self._index[1]) != len(res):
                    raise IndexError(
                        f'Op has {len(res)} outputs, but {len(self._index[1])} indices are given.'
                    )
                for i in range(len(res)):
                    setattr(arg[0], self._index[1][i], res[i])
                    arg[0].register(self._index[1][i])
            # Single output.
            else:
                setattr(arg[0], self._index[1], res)
                arg[0].register(self._index[1])

            return arg[0]
        else:
            res = self._op(*arg, **kws)
            return res

    def train(self, *arg, **kws):
        with self._lock:
            if self._op is None:
                self._op = op(self._name, self._tag, **self._kws)
        return self._op.train(*arg, **kws)

    def fit(self, *arg):
        self._op.fit(*arg)

    @property
    def is_stateful(self):
        with self._lock:
            if self._op is None:
                self._op = op(self._name, self._tag, **self._kws)
        return hasattr(self._op, 'fit')

    def set_state(self, state):
        with self._lock:
            if self._op is None:
                self._op = op(self._name, self._tag, **self._kws)
        self._op.set_state(state)

    @property
    def function(self):
        return self._name

    @property
    def init_args(self):
        return self._kws

    @staticmethod
    def callback(real_name: str, index: Tuple[str], *arg, **kws):
        if len(arg) == 0:
            return _OperatorLazyWrapper(real_name, index, **kws)
        else:
            return _OperatorLazyWrapper(real_name, index, arg[0], **kws)


ops = param_scope().callholder(_OperatorLazyWrapper.callback)
"""
Entry point for creating operator instances, for example:

>>> op_instance = ops.my_namespace.my_operator_name(init_arg1=xxx, init_arg2=xxx)

An instance of `my_namespace`/`my_operator_name` is created.
"""
