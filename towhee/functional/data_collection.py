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

from typing import Iterable, Iterator
from random import random, sample, shuffle

from towhee.hparam import param_scope


def _private_wrapper(func):

    def wrapper(self, *arg, **kws):
        create = DataCollection.cached
        if self.is_stream:
            create = DataCollection.stream

        return create(func(self, *arg, **kws))

    if hasattr(func, '__doc__'):  # pylint: disable=inconsistent-quotes
        wrapper.__doc__ = func.__doc__
    return wrapper


class DataCollection(Iterable):
    """
    DataCollection is a quick assambler for chained data processing operators.

    Examples:
    1. create a data collection from iterable
    >>> dc = DataCollection([1,2,3,4])

    2. chaining single line lambda operators:
    >>> result = dc.map(lambda x: x+1).map(lambda x: x*2)
    >>> list(result)
    [4, 6, 8, 10]

    3. chaining multiple line lambda operators
    scale supports chaining complex functions which is very convenient for
    data scientists to assemble a data processing pipeline.

    ```scala
    (1 to 100).map(x=>{
        y = x+1
        return 2 * y
    }).filter(x=>{
        y = x-1
        y < 4
    })
    ```
    Traditionally such pipeline is impossible in python,
    for multi-line lambda closure is not supported in python.
    DataCollection provides an alternative solution for
    chaining multi-line operators.

    chaining an operator with decorator syntax
    >>> dc = DataCollection([1,2,3,4])
    >>> @dc.map
    ... def add_1(x):
    ...     return x+1
    >>> @add_1.map
    ... def mul_2(x):
    ...     return x *2
    >>> list(mul_2)
    [4, 6, 8, 10]

    >>> dc = DataCollection([1,2,3,4])
    >>> @dc.filter
    ... def ge_3(x):
    ...     return x>=3
    >>> list(ge_3)
    [3, 4]

    4. dispatch chained call

    define operators
    >>> class my_add:
    ...     def __init__(self, val):
    ...         self.val = val
    ...     def __call__(self, x):
    ...         return x+self.val

    >>> class my_mul:
    ...     def __init__(self, val):
    ...         self.val = val
    ...     def __call__(self, x):
    ...         return x*self.val

    register the operators to data collection execution context
    >>> with param_scope(dispatcher={'add': my_add, 'mul': my_mul}):
    ...     dc = DataCollection([1,2,3,4])
    ...     result = dc.add(1).mul(2) # call registered operator
    >>> list(result)
    [4, 6, 8, 10]

    There are two kinds of data collection, streamed and cached.
    1. In a streamed data collection, the inputs are produced by upstream
    generator one by one. And operators have to process the input in a streamed
    manner. Streamed processing saves runtime memory, but some operations,
    such as indexing and shuffle, are not available.

    2. In a cached data collection, the inputs have to be loaded into the memory
    all at once, which requires huge memory if the data set is very large. But
    cached data collection allows indexing and shuffle.
    """

    def __init__(self, iterable) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def stream(arg):  # pylint: disable=no-self-argument
        """
        create a streamed data collection.

        Examples:
        1. create a streamed data collection
        >>> dc = DataCollection.stream([0,1,2,3,4])
        >>> dc.is_stream
        True

        2. convert a data collection to streamed version
        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc.is_stream
        False

        >>> dc = dc.stream()
        >>> dc.is_stream
        True
        """
        # pylint: disable=protected-access
        if isinstance(arg, DataCollection):
            self = arg
            if self.is_stream:
                return self
            return DataCollection.stream(self._iterable)

        iterable = arg
        if not isinstance(iterable, Iterator):
            iterable = iter(iterable)
        return DataCollection(iterable)

    def cached(arg):  # pylint: disable=no-self-argument
        """
        create a cached data collection.

        Examples:
        1. create a cached data collection
        >>> dc = DataCollection.cached(iter(range(5)))
        >>> dc.is_stream
        False

        2. convert a streamed data collection to cached version
        >>> dc = DataCollection(iter(range(5)))
        >>> dc.is_stream
        True
        >>> dc = dc.cached()
        >>> dc.is_stream
        False
        """
        # pylint: disable=protected-access
        if isinstance(arg, DataCollection):
            self = arg
            if not self.is_stream:
                return self
            return DataCollection.cached(self._iterable)

        iterable = arg
        if isinstance(iterable, Iterator):
            iterable = list(iterable)
        return DataCollection(iterable)

    @property
    def is_stream(self):
        """
        check whether the data collection is streamed.

        Examples:
        >>> dc = DataCollection([0,1,2,3,4])
        >>> dc.is_stream
        False

        >>> result = dc.map(lambda x: x+1)
        >>> result.is_stream
        False
        >>> result._iterable
        [1, 2, 3, 4, 5]

        >>> dc = DataCollection(iter(range(5)))
        >>> dc.is_stream
        True

        >>> result = dc.map(lambda x: x+1)
        >>> result.is_stream
        True
        >>> isinstance(result._iterable, Iterable)
        True

        """
        return isinstance(self._iterable, Iterator)

    def unstream(self):
        return self.cached()

    @_private_wrapper
    def map(self, unary_op):
        """
        apply operator to data collection
        >>> dc = DataCollection([1,2,3,4])
        >>> result = dc.map(lambda x: x+1).map(lambda x: x*2)
        >>> list(result)
        [4, 6, 8, 10]
        """
        return map(unary_op, self._iterable)

    @_private_wrapper
    def zip(self, other):
        """
        combine two data collections
        >>> dc1 = DataCollection([1,2,3,4])
        >>> dc2 = DataCollection([1,2,3,4]).map(lambda x: x+1)
        >>> dc3 = dc1.zip(dc2)
        >>> list(dc3)
        [(1, 2), (2, 3), (3, 4), (4, 5)]
        """
        return zip(self, other)

    @_private_wrapper
    def filter(self, unary_op):
        return filter(unary_op, self)

    @_private_wrapper
    def sample(self, ratio=1.0):
        """
        sample the data collection
        Examples:
        >>> dc = DataCollection(range(10000))
        >>> result = dc.sample(0.1)
        >>> 900 < len(list(result)) < 1100
        True
        """
        return filter(lambda _: random() < ratio, self)

    @staticmethod
    def range(*arg, **kws):
        """
        generate data collection with ranged numbers
        Examples:
        >>> dc = DataCollection.range(5)
        >>> list(dc)
        [0, 1, 2, 3, 4]
        """
        return DataCollection(range(*arg, **kws))

    @_private_wrapper
    def batch(self, size, drop_tail=False):
        """
        Create small batches from data collections
        Examples:
        >>> dc = DataCollection(range(10))
        >>> [list(batch) for batch in dc.batch(2)]
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        >>> dc = DataCollection(range(10))
        >>> [list(batch) for batch in dc.batch(3)]
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

        >>> dc = DataCollection(range(10))
        >>> [list(batch) for batch in dc.batch(3, drop_tail=True)]
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """

        def inner(iterable):
            buff = []
            for ele in iterable:
                buff.append(ele)
                if len(buff) == size:
                    yield DataCollection.cached(buff)
                    buff = []
            if not drop_tail and len(buff) > 0:
                yield DataCollection.cached(buff)

        return inner(self)

    @_private_wrapper
    def rolling(self, size, drop_head=True, drop_tail=True):
        """
        Create rolling windows from data collections.
        Examples:
        >>> dc = DataCollection(range(5))
        >>> [list(batch) for batch in dc.rolling(3)]
        [[0, 1, 2], [1, 2, 3], [2, 3, 4]]

        >>> dc = DataCollection(range(5))
        >>> [list(batch) for batch in dc.rolling(3, drop_head=False)]
        [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]

        >>> dc = DataCollection(range(5))
        >>> [list(batch) for batch in dc.rolling(3, drop_tail=False)]
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4], [4]]
        """

        def inner(iterable):
            buff = []
            for ele in iterable:
                buff.append(ele)
                if not drop_head or len(buff) == size:
                    yield DataCollection.cached(buff.copy())
                if len(buff) == size:
                    buff = buff[1:]
            while not drop_tail and len(buff) > 0:
                yield DataCollection.cached(buff)
                buff = buff[1:]

        return inner(self)

    @_private_wrapper
    def flaten(self):
        """
        flaten nested data collections

        >>> dc = DataCollection(range(10))
        >>> nested_dc = dc.batch(2)
        >>> list(nested_dc.flaten())
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """

        def inner(iterable):
            for ele in iterable:
                if isinstance(ele, Iterable):
                    for nested_ele in iter(ele):
                        yield nested_ele
                else:
                    yield ele

        return inner(self)

    def shuffle(self, in_place=False):
        """
        shuffle a cached data collection.

        Examples:
        1. shuffle
        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> shuffled = dc.shuffle()
        >>> tuple(dc) == tuple(range(5))
        True
        >>> tuple(shuffled) == tuple(range(5))
        False

        2. in place shuffle
        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> _ = dc.shuffle(True)
        >>> tuple(dc) == tuple(range(5))
        False

        3. streamed data collection is not supported
        >>> dc = DataCollection.stream([0, 1, 2, 3, 4])
        >>> _ = dc.shuffle()
        Traceback (most recent call last):
        TypeError: shuffle is not supported for streamed data collection.
        """
        if self.is_stream:
            raise TypeError(
                "shuffle is not supported for streamed data collection.")
        if in_place:
            shuffle(self._iterable)
            return self
        else:
            return DataCollection.cached(
                sample(self._iterable, len(self._iterable)))

    def __getattr__(self, name):
        with param_scope() as hp:
            dispatcher = hp().dispatcher({})
        op = dispatcher[name]

        def wrapper(*arg, **kws):
            op_instance = op(*arg, **kws)
            return self.map(op_instance)

        return wrapper

    def __getitem__(self, index):
        """
        indexing for data collection

        Examples:
        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc[0]
        0
        >>> dc.stream()[1]
        Traceback (most recent call last):
        TypeError: indexing is not supported for streamed data collection.
        """
        if self.is_stream:
            raise TypeError(
                "indexing is not supported for streamed data collection.")
        return self._iterable[index]

    def __setitem__(self, index, value):
        """
        indexing for data collection

        Examples:
        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc[0]
        0
        >>> dc[0] = 5
        >>> dc._iterable[0]
        5
        >>> dc.stream()[0]
        Traceback (most recent call last):
        TypeError: indexing is not supported for streamed data collection.
        """
        if self.is_stream:
            raise TypeError(
                "indexing is not supported for streamed data collection.")
        self._iterable[index] = value

    def __rshift__(self, unary_op):
        """
        chain the operators with `>>`

        Examples:
        >>> dc = DataCollection([1,2,3,4])
        >>> result = (dc
        ...     >> (lambda x: x+1)
        ...     >> (lambda x: x*2)
        ... )
        >>> list(result)
        [4, 6, 8, 10]
        """
        return self.map(unary_op)

    def __or__(self, unary_op):
        return self.map(unary_op)


if __name__ == '__main__':  # pylint: disable=inconsistent-quotes
    import doctest

    doctest.testmod(verbose=False)
