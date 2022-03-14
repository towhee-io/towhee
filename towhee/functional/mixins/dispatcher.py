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

from typing import Any

from towhee.engine.operator_loader import OperatorLoader


class _OperatorWrapper:

    def __init__(self, op, index):
        self._op = op
        self._index = index
        print(index)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._op(*args, **kwds)


class DispatcherMixin:
    """
    Mixin for call dispatcher for data collection

    >>> from towhee import register
    >>> from towhee import ops
    >>> from towhee.functional import DataCollection
    >>> @register('0.1', name='add_1')
    ... def add_1(x):
    ...     return x+1

    >>> dc = DataCollection.range(5).stream()
    >>> dc.add_1['a','b','c']()
    """

    def resolve(self, call_mapping, path, index, *arg, **kws):
        _ = index
        if path in call_mapping:
            return call_mapping[path](*arg, **kws)
        else:
            loader = OperatorLoader()
            op = loader.load_operator(path.replace('.', '/').replace('_', '-'),
                                      kws,
                                      tag='main')
            return _OperatorWrapper(op, index)

if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=False)
    # from towhee import register
    # from towhee import ops
    # from towhee.functional import DataCollection
    # @register('0.1', name='add_1')
    # def add_1(x):
    #     return x+1

    # dc = DataCollection.range(5).stream()
    # # dc.map(ops.add_1[('a','b','c')]())
    # print(ops.add_1)
    # print(ops.add_1['a']())
    # dc.add_1['a','b',('c', 'd')]()
    # # import ipdb; ipdb.set_trace()
