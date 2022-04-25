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

from typing import Any, Iterable, Iterator, Callable
from random import random, sample, shuffle
import reprlib
from towhee.functional.mixins.dag import close_dc, register_dag

from towhee.hparam import param_scope
from towhee.functional.option import Option, Some, Empty
from towhee.functional.mixins import AllMixins



class DataCollection(Iterable, AllMixins):
    """
    DataCollection is a pythonic computation and processing framework
    for unstructured data in machine learning and data science.
    It allows a data scientist or researcher to assemble a data processing pipeline,
    do his model work (embedding, transforming, or classification)
    and apply it to the business (search, recommendation, or shopping)
    with a method-chaining style API.

    Examples:

    1. Create a data collection from list or iterator:

    >>> dc = DataCollection([0, 1, 2, 3, 4])
    >>> dc = DataCollection(iter([0, 1, 2, 3, 4]))

    2. Chaining function invocations makes your code clean and fluent:

    >>> (
    ...    dc.map(lambda x: x+1)
    ...      .map(lambda x: x*2)
    ... ).to_list()
    [2, 4, 6, 8, 10]

    3. Multi-line closures are also supported via decorator syntax

    >>> dc = DataCollection([1,2,3,4])
    >>> @dc.map
    ... def add1(x):
    ...     return x+1
    >>> @add1.map
    ... def mul2(x):
    ...     return x *2
    >>> mul2.to_list()
    [4, 6, 8, 10]

    >>> dc = DataCollection([1,2,3,4])
    >>> @dc.filter
    ... def ge3(x):
    ...     return x>=3
    >>> ge3.to_list()
    [3, 4]

    `DataCollection` is designed to behave as a python list or iterator. Consider you are running
    the following code:

    .. code-block:: python
      :linenos:

      dc.map(stage1)
        .map(stage2)

    1. `iterator` and `stream mode`: When a `DataCollection` object is created from an iterator, it behaves as a python
    iterator and performs `stream-wise` data processing:

        a. `DataCollection` takes one element from the input and applies `stage1` and `stage2` sequentially ;
        b. Since DataCollection holds no data, indexing or shuffle is not supported;

    2. `list` and `unstream mode`: If a `DataCollection` object is created from a list, it will hold all the input values,
    and perform stage-wise computations:

        a. `stage2` will wait until all the calculations are done in `stage1`;
        b. A new DataCollection will be created to hold all the outputs for each stage. You can perform list operations on result DataCollection;

    """

    def __init__(self, iterable: Iterable) -> None:
        """Initializes a new DataCollection instance.

        Args:
            iterable (Iterable): input data
        """
        super().__init__()
        self._iterable = iterable

    def __iter__(self):
        if hasattr(self._iterable, 'iterrows'):
            return (x[1] for x in self._iterable.iterrows())
        return iter(self._iterable)

    def stream(arg):  # pylint: disable=no-self-argument
        """
        Create a stream data collection.

        Examples:

        1. Create a streamed data collection

        >>> dc = DataCollection.stream([0,1,2,3,4])
        >>> dc.is_stream
        True

        2. Convert a data collection to streamed version

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
            else:
                self._iterable = iter(self._iterable)
            return self

        iterable = arg
        if not isinstance(iterable, Iterator):
            iterable = iter(iterable)
        return DataCollection(iterable)

    def unstream(arg):  # pylint: disable=no-self-argument
        """
        Create a unstream data collection.

        Examples:

        1. Create a unstream data collection

        >>> dc = DataCollection.unstream(iter(range(5)))
        >>> dc.is_stream
        False

        2. Convert a streamed data collection to unstream version

        >>> dc = DataCollection(iter(range(5)))
        >>> dc.is_stream
        True
        >>> dc = dc.unstream()
        >>> dc.is_stream
        False
        """
        # pylint: disable=protected-access
        if isinstance(arg, DataCollection):
            self = arg
            if not self.is_stream:
                return self
            else:
                self._iterable = list(self._iterable)
            return self

        iterable = arg
        if isinstance(iterable, Iterator):
            iterable = list(iterable)
        return DataCollection(iterable)

    @property
    def is_stream(self):
        """
        Check whether the data collection is stream or unstream.

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
    def _factory(self):
        """
        Factory method for data collection.

        This factory method has been wrapped into a `param_scope()` which contains parent infomations.
        """
        creator = DataCollection.stream if self.is_stream else DataCollection.unstream
        def wrapper(*arg, **kws):
            with param_scope() as hp:
                hp().data_collection.parent = self
                x = creator(*arg, **kws)
                return x

        return wrapper

    def exception_safe(self):
        """
        Making the data collection exception-safe by warp elements with `Option`.

        Examples:

        1. Exception breaks pipeline execution:

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
        self._op = 'exception_safe'
        self._init_args = None
        self._call_args = None

        result = map(lambda x: Some(x) if not isinstance(x, Option) else x, self._iterable)
        return self._factory(result)

    def safe(self):
        """
        Shortcut for `exception_safe`
        """
        return self.exception_safe()


    def select_from(self, other):
        """
        Select data from dc with list(self).

        Examples:

        >>> dc1 = DataCollection([0.8, 0.9, 8.1, 9.2])
        >>> dc2 = DataCollection([[1, 2, 0], [2, 3, 0]])
        >>> dc3 = dc2.select_from(dc1)
        >>> list(dc3)
        [[0.9, 8.1, 0.8], [8.1, 9.2, 0.8]]
        """
        self._op = 'exception_safe'
        self._init_args = None
        self._call_args = other.id
        self.parent_id.append(other.id)
        other.notify_consumed(self._id)
 
        def inner(x):
            if isinstance(x, Iterable):
                return [other[i] for i in x]
            return other[x]

        result = map(inner, self._iterable)
        return self._factory(result)

    def fill_empty(self, default: Any = None) -> 'DataCollection':
        """
        Unbox `Option` values and fill `Empty` with default values.

        Args:
            default (Any): default value to replace empty values;

        Returns:
            DataCollection: data collection with empty values filled with `default`;

        Examples:

        >>> dc = DataCollection.range(5)
        >>> dc.safe().map(lambda x: x / (0 if x == 3 else 2)).fill_empty(-1.0).to_list()
        [0.0, 0.5, 1.0, -1.0, 2.0]
        """
        self._op = 'dc.fill_empty'
        result = map(lambda x: x.get() if isinstance(x, Some) else default, self._iterable)
        return self._factory(result)

    def drop_empty(self, callback: Callable = None) -> 'DataCollection':
        """
        Unbox `Option` values and drop `Empty`.

        Args:
            callback (Callable): handler for empty values;

        Returns:
            DataCollection: data collection that drops empty values;

        Examples:

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

    def map(self, *arg):
        """
        Apply operator to data collection.

        Args:
            *arg (Callable): functions/operators to apply to data collection;

        Returns:
            DataCollection: data collections that contains computation results;

        Examples:

        >>> dc = DataCollection([1,2,3,4])
        >>> dc.map(lambda x: x+1).map(lambda x: x*2).to_list()
        [4, 6, 8, 10]
        """

        # mmap
        if len(arg) > 1:
            return self.mmap(list(arg))

        unary_op = arg[0]

        # smap map for stateful operator
        if hasattr(unary_op, 'is_stateful') and unary_op.is_stateful:
            return self.smap(unary_op)

        # pmap
        if self.executor is not None:
            return self.pmap(unary_op)

        if hasattr(self._iterable, 'map'):
            return self._factory(self._iterable.map(unary_op))

        if hasattr(self._iterable, 'apply') and hasattr(unary_op, '__dataframe_apply__'):
            return self._factory(unary_op.__dataframe_apply__(self._iterable))

        # map
        def inner(x):
            if isinstance(x, Option):
                return x.map(unary_op)
            else:
                return unary_op(x)

        result = map(inner, self._iterable)
        return self._factory(result)

    @register_dag
    def zip(self, *others) -> 'DataCollection':
        """
        Combine two data collections.

        Args:
            *others (DataCollection): other data collections;

        Returns:
            DataCollection: data collection with zipped values;

        Examples:

        >>> dc1 = DataCollection([1,2,3,4])
        >>> dc2 = DataCollection([1,2,3,4]).map(lambda x: x+1)
        >>> dc3 = dc1.zip(dc2)
        >>> list(dc3)
        [(1, 2), (2, 3), (3, 4), (4, 5)]
        """
        self._op = 'zip'
        self._init_args = None
        # self._call_args = [other.id for other in others]
        # self._parent_id.extend([other.id for other in others])
        # print(self._parent_id)
        
        for x in others:
            x.notify_consumed(self._id)

        return self._factory(zip(self, *others))

    def filter(self, unary_op: Callable, drop_empty=False) -> 'DataCollection':
        """
        Filter data collection with `unary_op`.

        Args:
            unary_op (`Callable`):
                Callable to decide whether to filter the element;
            drop_empty (`bool`):
                Drop empty values. Defaults to False.

        Returns:
            DataCollection: filtered data collection
        """
        self._dag[self._id] = ('filter', (unary_op, drop_empty),[])

        # return filter(unary_op, self)
        def inner(x):
            if isinstance(x, Option):
                if isinstance(x, Some):
                    return unary_op(x.get())
                return not drop_empty
            return unary_op(x)

        if hasattr(self._iterable, 'filter'):
            return self._factory(self._iterable.filter(unary_op))

        if hasattr(self._iterable, 'apply') and hasattr(unary_op, '__dataframe_filter__'):
            return DataCollection(unary_op.__dataframe_apply__(self._iterable))

        return self._factory(filter(inner, self._iterable))

    def sample(self, ratio=1.0) -> 'DataCollection':
        """
        Sample the data collection.

        Args:
            ratio (float): sample ratio;

        Returns:
            DataCollection: sampled data collection;

        Examples:

        >>> dc = DataCollection(range(10000))
        >>> result = dc.sample(0.1)
        >>> ratio = len(result.to_list()) / 10000.
        >>> 0.09 < ratio < 0.11
        True
        """
        self._dag[self._id] = ('sample', (), [])
        return self._factory(filter(lambda _: random() < ratio, self))

    @staticmethod
    def range(*arg, **kws):
        """
        Generate data collection with ranged numbers.

        Examples:

        >>> DataCollection.range(5).to_list()
        [0, 1, 2, 3, 4]
        """
        return DataCollection(range(*arg, **kws))

    def batch(self, size, drop_tail=False) -> 'DataCollection':
        """
        Create small batches from data collections.

        Args:
            size (int): window size;
            drop_tail (bool): drop tailing windows that not full;

        Returns:
            DataCollection: data collection of batched windows;

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
        self._dag[self._id] = ('batch', (size, drop_tail),[])
        def inner():
            buff = []
            for ele in self._iterable:
                buff.append(ele)
                if len(buff) == size:
                    yield DataCollection.unstream(buff)
                    buff = []
            if not drop_tail and len(buff) > 0:
                yield DataCollection.unstream(buff)

        return self._factory(inner())

    def rolling(self, size: int, drop_head=True, drop_tail=True):
        """
        Create rolling windows from data collections.

        Args:
            size (int): window size;
            drop_head (bool): drop headding windows that not full;
            drop_tail (bool): drop tailing windows that not full;

        Returns:
            DataCollection: data collection of rolling windows;

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
        self._dag[self._id] = ('rolling', (), [])
        def inner():
            buff = []
            for ele in self._iterable:
                buff.append(ele)
                if not drop_head or len(buff) == size:
                    yield DataCollection.unstream(buff.copy())
                if len(buff) == size:
                    buff = buff[1:]
            while not drop_tail and len(buff) > 0:
                yield DataCollection.unstream(buff)
                buff = buff[1:]

        return self._factory(inner())

    def flatten(self) -> 'DataCollection':
        """
        Flatten nested data collections.

        Returns:
            DataCollection: flattened data collection;

        Examples:

        >>> dc = DataCollection(range(10))
        >>> nested_dc = dc.batch(2)
        >>> nested_dc.flatten().to_list()
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        self._dag[self._id] = ('flatten', (), [])
        def inner():
            for ele in self._iterable:
                if isinstance(ele, Iterable):
                    for nested_ele in iter(ele):
                        yield nested_ele
                else:
                    yield ele

        return self._factory(inner())

    def shuffle(self, in_place=False) -> 'DataCollection':
        """
        Shuffle an unstreamed data collection.

        Args:
            in_place (bool): shuffle the data collection in place;

        Returns:
            DataCollection: shuffled data collection;

        Examples:

        1. Shuffle:

        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> shuffled = dc.shuffle()
        >>> tuple(dc) == tuple(range(5))
        True

        >>> tuple(shuffled) == tuple(range(5))
        False

        2. In place shuffle:

        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> _ = dc.shuffle(True)
        >>> tuple(dc) == tuple(range(5))
        False

        3. streamed data collection is not supported:

        >>> dc = DataCollection.stream([0, 1, 2, 3, 4])
        >>> _ = dc.shuffle()
        Traceback (most recent call last):
        TypeError: shuffle is not supported for streamed data collection.
        """
        self._dag[self._id] = ('shuffle', (), [])
        if self.is_stream:
            raise TypeError('shuffle is not supported for streamed data collection.')
        if in_place:
            shuffle(self._iterable)
            return self
        else:
            return DataCollection.unstream(sample(self._iterable, len(self._iterable)))

    def __getattr__(self, name):
        """
        Unknown method dispatcher.

        When a unknown method is invoked on a `DataCollection` object,
        the function call will be dispatched to a method resolver.
        By registering function to the resolver, you are able to extend
        `DataCollection`'s API at runtime without modifying its code.

        Examples:

        1. Define two operators:

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

        2. Register the operators to `DataCollection`'s execution context with `param_scope`:

        >>> from towhee import param_scope
        >>> with param_scope(dispatcher={
        ...         'add': my_add, # register `my_add` as `dc.add`
        ...         'mul': my_mul  # register `my_mul` as `dc.mul`
        ... }):
        ...     dc = DataCollection([1,2,3,4])
        ...     dc.add(1).mul(2).to_list() # call registered operator
        [4, 6, 8, 10]
        """
        with param_scope() as hp:
            dispatcher = hp().dispatcher({})

            def wrapper(path, index, *arg, **kws):
                _ = index
                op = self.resolve(dispatcher, path, index, *arg, **kws)
                return self.map(op)

            callholder = hp.callholder(wrapper)
        return getattr(callholder, name)

    def __getitem__(self, index):
        """
        Indexing for data collection.

        Examples:

        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc[0]
        0

        >>> dc.stream()[1]
        Traceback (most recent call last):
        TypeError: indexing is only supported for data collection created from list or pandas DataFrame.
        """
        if not hasattr(self._iterable, '__getitem__'):
            raise TypeError('indexing is only supported for '
                            'data collection created from list or pandas DataFrame.')
        return self._iterable[index]

    def __setitem__(self, index, value):
        """
        Indexing for data collection.

        Examples:

        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc[0]
        0

        >>> dc[0] = 5
        >>> dc._iterable[0]
        5

        >>> dc.stream()[0]
        Traceback (most recent call last):
        TypeError: indexing is only supported for data collection created from list or pandas DataFrame.
        """
        if not hasattr(self._iterable, '__setitem__'):
            raise TypeError('indexing is only supported for '
                            'data collection created from list or pandas DataFrame.')
        self._iterable[index] = value

    def __rshift__(self, unary_op):
        """
        Chain the operators with `>>`.

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

    def __add__(self, other):
        """
        Concat two data collections:

        Examples:

        >>> (DataCollection.range(5) + DataCollection.range(5)).to_list()
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

        >>> (DataCollection.range(5) + DataCollection.range(5) + DataCollection.range(5)).to_list()
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        """
        #TODO Needs a better solution for combining branches
        swap_id = other._id
        for values in self._dag.values():
            temp = [x if x is not swap_id else self._id for x in values[2]]
            values[2].clear()
            values[2].extend(temp)
        self._dag[self._id] = ('add', (other._id,) ,[])
        def inner():
            for x in self:
                yield x
            for x in other:
                yield x

        return self._factory(inner())

    def __repr__(self) -> str:
        """
        Return a string representation for DataCollection.

        Examples:

        >>> DataCollection.unstream([1, 2, 3])
        [1, 2, 3]

        >>> DataCollection.stream([1, 2, 3]) #doctest: +ELLIPSIS
        <list_iterator object at...>
        """
        if isinstance(self._iterable, list):
            return reprlib.repr(self._iterable)
        if hasattr(self._iterable, '__repr__'):
            return repr(self._iterable)
        return super().__repr__()

    def head(self, n: int = 5):
        """
        Get the first n lines of a DataCollection.

        Args:
            n (`int`):
                The number of lines to print. Default value is 5.

        Examples:

        >>> DataCollection.range(10).head(3).to_list()
        [0, 1, 2]
        """
        self._dag[self._id] = ('head', (n,), [])
        def inner():
            for i, x in enumerate(self._iterable):
                if i >= n:
                    break
                yield x
        return self._factory(inner())

    @close_dc
    def to_list(self):
        return self._iterable if isinstance(self._iterable, list) else list(self)


if __name__ == '__main__':  # pylint: disable=inconsistent-quotes
    import doctest

    doctest.testmod(verbose=False)
