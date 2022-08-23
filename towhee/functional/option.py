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
    """The reason for the current value.

        Args:
            x (Any): The value.
            e (Exception): The Exception that caused it.
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
    """Functional-style error handling.

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
        """Return a boxed value.

        Args:
            x (T): The input value.

        Returns:
            Some(T): The boxed value.
        """
        return Some(x)

    @staticmethod
    def empty():
        """Return an empty value.

        Returns:
            Empty: The empty value.
        """
        return Empty()

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        """Apply boxed version of callable.

        Args:
            f (Callable[[A], Option[B]]): Boxed version of callable

        Returns:
            Option[B]: The boxed value.

        Examples:

            >>> Option.of(1).flat_map(lambda x: x+1)
            2

            >>> Option.empty().flat_map(lambda x: x+1)
            Empty()
        """
        if isinstance(self, Some):
            return f(self._value)
        else:
            return self

    def map(self, f: Callable[[A], 'B']) -> 'Option[B]':
        """Apply a function to a value.

        Args:
            f (Callable[[A], B]): A unboxed callable.

        Returns:
            Option[B]: The boxed return value.
        """

        def wrapper(x):
            try:
                return Some(f(x))
            except Exception as e:  # pylint: disable=broad-except
                return Empty(x, e)

        return self.flat_map(wrapper)

    def is_empty(self):
        """Return True if the value is empty.

        Returns:
            bool: True if empty.
        """
        return isinstance(self, Empty)

    def is_some(self):
        """Return True if the value is boxed value.

        Returns:
            bool: True if boxed.
        """
        return isinstance(self, Some)

    def get_or_else(self, default):
        """Return unboxed value, or default if the value is empty.

        Args:
            default (any): The default value to return.

        Returns:
            any: The unboxed value, or default if empty.

        Examples:

            >>> Option.of(0).get_or_else(1)
            0

            >>> Option.empty().get_or_else(1)
            1
        """
        if self.is_some():
            return self.get()
        return default


class Some(Option[A]):
    """`Some` value for `Option`
    """

    def __init__(self, x: A) -> None:
        self._value = x

    def __repr__(self) -> str:
        return 'Some({})'.format(self._value)

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        return f(self._value)

    def get(self):
        """Return unboxed value
        """
        return self._value


class Empty(Option[A]):
    """`Empty` value for `Option`

        Args:
            x (Any, optional): The value. Defaults to None.
            e (Exception, optional): The reasone for value. Defaults to None.
    """
    def __init__(self, x: Any = None, e: Exception = None) -> None:
        self._reason = _Reason(x, e)

    def __repr__(self) -> str:
        """The repr function that returns default 'Empty()'.

        Returns:
            str: Returns 'Empty()'.
        """
        return 'Empty()'

    def flat_map(self, f: Callable[[A], 'Option[B]']) -> 'Option[B]':
        return self

    def get(self):
        """Return the reason of the empty value.

        Returns:
            _Reason: Reason for empty.
        """
        return self._reason


empty = Empty()
