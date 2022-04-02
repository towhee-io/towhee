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
from typing import Callable

from towhee.engine import register

# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name

@register(name='builtin/runas_op')
class runas_op:
    """
    Convert a user-defined function as an operator and execute.

    Args:
        func (`Callable`):
            The user-defined function.

    Examples:

    >>> from towhee.functional import DataCollection
    >>> from towhee.functional.entity import Entity
    >>> entities = [Entity(a=i, b=i) for i in range(5)]
    >>> dc = DataCollection(entities)
    >>> res = dc.runas_op['a', 'b'](func=lambda x: x - 1).to_list()
    >>> res[0].a = res[0].b + 1
    True
    """
    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, *args, **kws):
        return self._func(*args, **kws)


# __test__ = {'run': run.__doc__}

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)
