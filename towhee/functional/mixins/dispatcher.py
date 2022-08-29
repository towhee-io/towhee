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

from towhee.engine.factory import ops, create_op
from towhee.hparam import param_scope


class DispatcherMixin:
    """Mixin for call dispatcher for data collection.

    >>> import towhee
    >>> from towhee import register
    >>> @register(name='add_1')
    ... def add_1(x):
    ...     return x+1

    >>> dc = towhee.range(5).stream()
    >>> dc.add_1['a','b','c']() #doctest: +ELLIPSIS
    <map object at ...>
    """

    def resolve(self, path, index, *arg, **kws):
        """Dispatch unknown operators.

        Args:
            path (str): The operator name.
            index (str): The index of data being called on.

        Returns:
            _OperatorLazyWrapper: The operator that corresponds to the path.
        """
        with param_scope() as hp:
            locals_ = hp.locals
            globals_ = hp.globals
            op = None
            if '.' not in path:
                if path in locals_:
                    op = locals_[path]
                elif path in globals_:
                    op = globals_[path]
            if op is not None and callable(op):
                if isinstance(op, type):
                    instance = op(*arg, **kws)
                else:
                    instance = op
                return create_op(instance, path, index, arg, kws)
            return getattr(ops, path)[index](*arg, **kws)
