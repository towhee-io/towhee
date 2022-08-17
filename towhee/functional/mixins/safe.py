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
from typing import Any, Callable

from towhee.functional.option import Empty
from towhee.functional.mixins.dag import register_dag
from towhee.functional.option import Option, Some


class SafeMixin:
    """
    Mixin for exception safety.
    """
    @register_dag
    def exception_safe(self):
        """
        Making the data collection exception-safe by warp elements with `Option`.

        Examples:

        1. Exception breaks pipeline execution:

        >>> from towhee import DataCollection
        >>> dc = DataCollection.range(5)
        >>> dc.map(lambda x: x / (0 if x == 3 else 2)).to_list()
        Traceback (most recent call last):
        ZeroDivisionError: division by zero

        2. Exception-safe execution

        >>> dc.exception_safe().map(lambda x: x / (0 if x == 3 else 2)).to_list()
        [Some(0.0), Some(0.5), Some(1.0), Empty(), Some(2.0)]

        >>> dc.exception_safe().map(lambda x: x / (0 if x == 3 else 2)).filter(lambda x: x < 1.5).to_list()
        [Some(0.0), Some(0.5), Some(1.0), Empty()]

        >>> dc.exception_safe().map(lambda x: x / (0 if x == 3 else 2)).filter(lambda x: x < 1.5, drop_empty=True).to_list()
        [Some(0.0), Some(0.5), Some(1.0)]
        """
        result = map(lambda x: Some(x) if not isinstance(x, Option) else x, self._iterable)
        return self._factory(result)

    def safe(self):
        """
        Shortcut for `exception_safe`
        """
        return self.exception_safe()

    @register_dag
    def fill_empty(self, default: Any = None) -> 'DataCollection':
        """
        Unbox `Option` values and fill `Empty` with default values.

        Args:
            default (Any): default value to replace empty values;

        Returns:
            DataCollection: data collection with empty values filled with `default`;

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection.range(5)
        >>> dc.safe().map(lambda x: x / (0 if x == 3 else 2)).fill_empty(-1.0).to_list()
        [0.0, 0.5, 1.0, -1.0, 2.0]
        """
        result = map(lambda x: x.get() if isinstance(x, Some) else default, self._iterable)
        return self._factory(result)

    @register_dag
    def drop_empty(self, callback: Callable = None) -> 'DataCollection':
        """
        Unbox `Option` values and drop `Empty`.

        Args:
            callback (Callable): handler for empty values;

        Returns:
            DataCollection: data collection that drops empty values;

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection.range(5)
        >>> dc.safe().map(lambda x: x / (0 if x == 3 else 2)).drop_empty().to_list()
        [0.0, 0.5, 1.0, 2.0]

        Get inputs that case exceptions:

        >>> exception_inputs = []
        >>> result = dc.safe().map(lambda x: x / (0 if x == 3 else 2)).drop_empty(lambda x: exception_inputs.append(x.get().value))
        >>> exception_inputs
        [3]
        """
        if callback is not None:

            def inner(data):
                for x in data:
                    if isinstance(x, Empty):
                        callback(x)
                    if isinstance(x, Some):
                        yield x.get()

            result = inner(self._iterable)
        else:

            def inner(data):
                for x in data:
                    if isinstance(x, Some):
                        yield x.get()

            result = inner(self._iterable)
        return self._factory(result)

