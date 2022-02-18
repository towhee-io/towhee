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
from typing import Callable, Generic
from typing import Any, TypeVar

A = TypeVar('A', covariant=True)
B = TypeVar('B')
T = TypeVar('T')


class _Reason:
    """
    reason for `Empty` value
    """
    def __init__(self, x: Any, e: Exception) -> None:
        self._value = x
        self._exception = e

    @property
    def value(self):
        return self._value

    @property
    def exception(self):
        return self._exception


class Option(Generic[A]):
    """
    Functional-style error handling.

    Option[A] = Some(A) or Empty[A]
    1. Some(A): just a container for result;
    2. Empty[A]: result is empty, because of input error or computation error;

    Examples:
    >>> a = Some(10)
    >>> a.map(lambda x: x/2.0)
    Some(5.0)

    >>> a.map(lambda x: x/0)
    Empty()

    >>> b = Empty()
    >>> b.map(lambda x: x/2.0)
    Empty()
    """

    @staticmethod
    def of(x: T):
        return Some(x)

    @staticmethod
    def empty():
        return Empty()

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        if isinstance(self, Some):
            return f(self._value)
        else:
            return self

    def map(self, f: Callable[[A], 'B']) -> 'Option[B]':

        def wrapper(x):
            try:
                return Some(f(x))
            except Exception as e: # pylint: disable=broad-except
                return Empty(_Reason(x, e))

        return self.flat_map(wrapper)

    def is_empty(self):
        return isinstance(self, Empty)

    def is_some(self):
        return isinstance(self, Some)

    def get_or_else(self, default):
        if self.is_some():
            return self.get()
        return default


class Some(Option[A]):
    """
    `Some` value for `Option`
    """
    def __init__(self, x: A) -> None:
        self._value = x

    def __repr__(self) -> str:
        return 'Some({})'.format(self._value)

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        return f(self._value)

    def get(self):
        return self._value


class Empty(Option[A]):
    """
    `Empty` value for `Option`
    """
    def __init__(self, reason=_Reason(None, None)) -> None:
        self._reason = reason

    def __repr__(self) -> str:
        return 'Empty()'

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        return self

    def get(self):
        return self._reason


empty = Empty()

if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=False)
