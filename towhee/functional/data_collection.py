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
from towhee.functional.option import Option, Some, Empty

from towhee.functional.mixins.data_source import DataSourceMixin
from towhee.functional.mixins.dispatcher import DispatcherMixin
from towhee.functional.mixins.parallel import ParallelMixin
from towhee.functional.mixins.computer_vision import ComputerVisionMixin


def _private_wrapper(func):

    def wrapper(self, *arg, **kws):
        return self.factory(func(self, *arg, **kws))

    if hasattr(func, '__doc__'):  # pylint: disable=inconsistent-quotes
        wrapper.__doc__ = func.__doc__
    return wrapper


class DataCollection(Iterable, DataSourceMixin, DispatcherMixin, ParallelMixin,
                     ComputerVisionMixin):
    """
    DataCollection is a quick assambler for chained data processing operators.

    Examples:
    1. create a data collection from iterable
    >>> dc = DataCollection([1,2,3,4])

    2. chaining single line lambda operators:
    >>> dc.map(lambda x: x+1).map(lambda x: x*2).to_list()
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
    ...     dc.add(1).mul(2).to_list() # call registered operator
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
        super().__init__()
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

    def unstream(self):
        return self.cached()

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

    @property
    def factory(self):
        """
        Factory method for data collection.

        This factory method has been wrapped into a `param_scope()` which contains parent infomations.
        """
        creator = DataCollection.stream if self.is_stream else DataCollection.cached

        def wrapper(*arg, **kws):
            with param_scope() as hp:
                hp().data_collection.parent = self
                return creator(*arg, **kws)

        return wrapper

    @_private_wrapper
    def exception_safe(self):
        """
        making the data collection exception-safe by warp elements with `Option`.

        Examples:
        1. exception breaks pipeline execution:
        >>> dc = DataCollection.range(5)
        >>> dc.map(lambda x: x / (0 if x == 3 else 2)).to_list()
        Traceback (most recent call last):
        ZeroDivisionError: division by zero

        2. exception-safe execution
        >>> dc.exception_safe().map(lambda x: x / (0 if x == 3 else 2)).to_list()
        [Some(0.0), Some(0.5), Some(1.0), Empty(), Some(2.0)]

        >>> dc.exception_safe().map(lambda x: x / (0 if x == 3 else 2)).filter(lambda x: x < 1.5).to_list()
        [Some(0.0), Some(0.5), Some(1.0), Empty()]

        >>> dc.exception_safe().map(lambda x: x / (0 if x == 3 else 2)).filter(lambda x: x < 1.5, drop_empty=True).to_list()
        [Some(0.0), Some(0.5), Some(1.0)]
        """
        return map(lambda x: Some(x)
                   if not isinstance(x, Option) else x, self._iterable)

    def safe(self):
        """
        shortcut for `exception_safe`
        """
        return self.exception_safe()

    def select(self, name: str = 'feature_vector'):
        """
        get the list from operator Output

        Examples:
        >>> from typing import NamedTuple
        >>> Outputs = NamedTuple('Outputs', [('num', int)])
        >>> dc = DataCollection([Outputs(1), Outputs(2), Outputs(3)])
        >>> dc.select(name='num')
        [1, 2, 3]
        """
        return [getattr(i, name) for i in self._iterable]

    @_private_wrapper
    def select_from(self, dc):
        """
        select data from dc with list(self)
        >>> dc1 = DataCollection([0.8, 0.9, 8.1, 9.2])
        >>> dc2 = DataCollection([[1, 2, 0], [2, 3, 0]])
        >>> dc3 = dc1.select_from(dc2)
        >>> list(dc3)
        [[0.9, 8.1, 0.8], [8.1, 9.2, 0.8]]
        """
        dc_list = dc.to_list()
        ids = self.to_list()
        res = []
        for sel_id in ids:
            res.append([dc_list[k] for k in sel_id])
        return res

    @_private_wrapper
    def fill_empty(self, default=None):
        """
        unbox `Option` values and fill `Empty` with default values

        Examples:
        >>> dc = DataCollection.range(5)
        >>> dc.safe().map(lambda x: x / (0 if x == 3 else 2)).fill_empty(-1.0).to_list()
        [0.0, 0.5, 1.0, -1.0, 2.0]
        """
        return map(lambda x: x.get()
                   if isinstance(x, Some) else default, self._iterable)

    @_private_wrapper
    def drop_empty(self, callback=None):
        """
        unbox `Option` values and drop `Empty`

        Examples:
        >>> dc = DataCollection.range(5)
        >>> dc.safe().map(lambda x: x / (0 if x == 3 else 2)).drop_empty().to_list()
        [0.0, 0.5, 1.0, 2.0]

        get inputs that case exceptions:
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

            return inner(self._iterable)
        else:

            def inner(data):
                for x in data:
                    if isinstance(x, Some):
                        yield x.get()

            return inner(self._iterable)

    @_private_wrapper
    def map(self, *arg):
        """
        apply operator to data collection

        Examples:
        >>> dc = DataCollection([1,2,3,4])
        >>> dc.map(lambda x: x+1).map(lambda x: x*2).to_list()
        [4, 6, 8, 10]
        """

        # return map(unary_op, self._iterable)
        # mmap
        if len(arg) > 1:
            return self.mmap(*arg)
        unary_op = arg[0]

        # pmap
        if self.get_executor() is not None:
            return self.pmap(unary_op, executor=self._executor)

        #map
        def inner(x):
            if isinstance(x, Option):
                return x.map(unary_op)
            else:
                return unary_op(x)

        return map(inner, self._iterable)

    @_private_wrapper
    def zip(self, *others):
        """
        combine two data collections
        >>> dc1 = DataCollection([1,2,3,4])
        >>> dc2 = DataCollection([1,2,3,4]).map(lambda x: x+1)
        >>> dc3 = dc1.zip(dc2)
        >>> list(dc3)
        [(1, 2), (2, 3), (3, 4), (4, 5)]
        """
        return zip(self, *others)

    @_private_wrapper
    def filter(self, unary_op, drop_empty=False):
        #return filter(unary_op, self)
        def inner(x):
            if isinstance(x, Option):
                if isinstance(x, Some):
                    return unary_op(x.get())
                return not drop_empty
            return unary_op(x)

        return filter(inner, self._iterable)

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
        >>> DataCollection.range(5).to_list()
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
        buff = []
        for ele in self._iterable:
            buff.append(ele)
            if len(buff) == size:
                yield DataCollection.cached(buff)
                buff = []
        if not drop_tail and len(buff) > 0:
            yield DataCollection.cached(buff)

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

        buff = []
        for ele in self._iterable:
            buff.append(ele)
            if not drop_head or len(buff) == size:
                yield DataCollection.cached(buff.copy())
            if len(buff) == size:
                buff = buff[1:]
        while not drop_tail and len(buff) > 0:
            yield DataCollection.cached(buff)
            buff = buff[1:]

    @_private_wrapper
    def flaten(self):
        """
        flaten nested data collections

        >>> dc = DataCollection(range(10))
        >>> nested_dc = dc.batch(2)
        >>> nested_dc.flaten().to_list()
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """

        for ele in self._iterable:
            if isinstance(ele, Iterable):
                for nested_ele in iter(ele):
                    yield nested_ele
            else:
                yield ele

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
                'shuffle is not supported for streamed data collection.')
        if in_place:
            shuffle(self._iterable)
            return self
        else:
            return DataCollection.cached(
                sample(self._iterable, len(self._iterable)))

    def __getattr__(self, name):
        """
        Call dispatcher for data collection
        """
        with param_scope() as hp:
            dispatcher = hp().dispatcher({})

            def wrapper(path, *arg, **kws):
                op = self.resolve(dispatcher, path, *arg, **kws)
                return self.map(op)

            callholder = hp.callholder(wrapper)
        return getattr(callholder, name)

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
                'indexing is not supported for streamed data collection.')
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
                'indexing is not supported for streamed data collection.')
        self._iterable[index] = value

    def __rshift__(self, unary_op):
        """
        chain the operators with `>>`

        Examples:
        >>> dc = DataCollection([1,2,3,4])
        >>> (dc
        ...     >> (lambda x: x+1)
        ...     >> (lambda x: x*2)
        ... ).to_list()
        [4, 6, 8, 10]
        """
        return self.map(unary_op)

    def __or__(self, unary_op):
        return self.map(unary_op)

    @_private_wrapper
    def __add__(self, other):
        """
        concat two data collection.

        Examples:
        >>> (DataCollection.range(5) + DataCollection.range(5)).to_list()
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        >>> (DataCollection.range(5) + DataCollection.range(5) + DataCollection.range(5)).to_list()
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        """

        for x in self:
            yield x
        for x in other:
            yield x

    def to_list(self):
        return self._iterable if isinstance(self._iterable,
                                            list) else list(self)


if __name__ == '__main__':  # pylint: disable=inconsistent-quotes
    import doctest

    doctest.testmod(verbose=False)
