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
import random
import reprlib

from towhee.functional.mixins.dag import register_dag
from towhee.hparam import param_scope, dynamic_dispatch
from towhee.functional.entity import Entity, EntityView
from towhee.functional.option import Option, Some, Empty
from towhee.functional.mixins import DCMixins
from towhee.functional.mixins.dataframe import DataFrameMixin
from towhee.functional.mixins.column import ColumnMixin


class DataCollection(Iterable, DCMixins):
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

    @register_dag
    def stream(self):
        """
        Create a stream data collection.

        Examples:
        1. Convert a data collection to streamed version

        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc.is_stream
        False

        >>> dc = dc.stream()
        >>> dc.is_stream
        True
        """
        # pylint: disable=protected-access
        iterable = iter(self._iterable) if not self.is_stream else self._iterable
        return self._factory(iterable, parent_stream=False)

    @register_dag
    def unstream(self):
        """
        Create a unstream data collection.

        Examples:

        1. Create a unstream data collection

        >>> dc = DataCollection(iter(range(5))).unstream()
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
        iterable = list(self._iterable) if self.is_stream else self._iterable
        return self._factory(iterable, parent_stream=False)

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

    def _factory(self, iterable, parent_stream=True):
        """
        Factory method for data collection.

        This factory method has been wrapped into a `param_scope()` which contains parent information.

        Args:
            iterable: An iterable object, the data being stored in the DC
            parent_stream: Whether to copy the parents format (streamed vs unstreamed)

        Returns:
            DataCollection: DataCollection encapsulating the iterable.
        """
        if parent_stream is True:
            if self.is_stream:
                if not isinstance(iterable, Iterator):
                    iterable = iter(iterable)
            else:
                if isinstance(iterable, Iterator):
                    iterable = list(iterable)

        with param_scope() as hp:
            hp().data_collection.parent = self
            return DataCollection(iterable)

    @register_dag
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
        result = map(lambda x: Some(x) if not isinstance(x, Option) else x, self._iterable)
        return self._factory(result)

    def safe(self):
        """
        Shortcut for `exception_safe`
        """
        return self.exception_safe()

    @register_dag
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
        self.parent_ids.append(other.id)
        other.notify_consumed(self.id)

        def inner(x):
            if isinstance(x, Iterable):
                return [other[i] for i in x]
            return other[x]

        result = map(inner, self._iterable)
        return self._factory(result)

    @register_dag
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

    @register_dag
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
        if self.get_executor() is not None:
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
        self.parent_ids.extend([other.id for other in others])

        for x in others:
            x.notify_consumed(self.id)

        return self._factory(zip(self, *others))

    @register_dag
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

    @register_dag
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
        return self._factory(filter(lambda _: random.random() < ratio, self))

    @staticmethod
    @register_dag
    def range(*arg, **kws):
        """
        Generate data collection with ranged numbers.

        Examples:

        >>> DataCollection.range(5).to_list()
        [0, 1, 2, 3, 4]
        """
        return DataCollection(range(*arg, **kws))

    @register_dag
    def batch(self, size, drop_tail=False, raw=True):
        """
        Create small batches from data collections.

        Args:
            size (int): window size;
            drop_tail (bool): drop tailing windows that not full, defaults to False;
            raw (bool): whether to return raw data instead of DataCollection, defaults to True

        Returns:
            DataCollection of batched windows or batch raw data

        Examples:

        >>> dc = DataCollection(range(10))
        >>> [list(batch) for batch in dc.batch(2, raw=False)]
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        >>> dc = DataCollection(range(10))
        >>> dc.batch(3)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

        >>> dc = DataCollection(range(10))
        >>> dc.batch(3, drop_tail=True)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        >>> from towhee import Entity
        >>> dc = DataCollection([Entity(a=a, b=b) for a,b in zip(['abc', 'vdfvcd', 'cdsc'], [1,2,3])])
        >>> dc.batch(2)
        [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]
        """

        def inner():
            buff = []
            count = 0
            for ele in self._iterable:
                if isinstance(ele, Entity):
                    if count == 0:
                        buff = ele
                        for key in ele.__dict__.keys():
                            buff.__dict__[key] = [buff.__dict__[key]]
                        count = 1
                        continue
                    for key in ele.__dict__.keys():
                        buff.__dict__[key].append(ele.__dict__[key])
                else:
                    buff.append(ele)
                count += 1

                if count == size:
                    if raw:
                        yield buff
                    else:
                        yield buff if isinstance(buff, list) else [buff]
                    buff = []
                    count = 0
            if not drop_tail and count > 0:
                if raw:
                    yield buff
                else:
                    yield buff if isinstance(buff, list) else [buff]

        return self._factory(inner())

    @register_dag
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

        def inner():
            buff = []
            for ele in self._iterable:
                buff.append(ele)
                if not drop_head or len(buff) == size:
                    yield buff.copy()
                if len(buff) == size:
                    buff = buff[1:]
            while not drop_tail and len(buff) > 0:
                yield buff
                buff = buff[1:]

        return self._factory(inner())

    @register_dag
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

        def inner():
            for ele in self._iterable:
                if isinstance(ele, Iterable):
                    for nested_ele in iter(ele):
                        yield nested_ele
                else:
                    yield ele

        return self._factory(inner())

    @register_dag
    def shuffle(self) -> 'DataCollection':
        """
        Shuffle an unstreamed data collection in place.

        Returns:
            DataCollection: shuffled data collection;

        Examples:

        1. Shuffle:

        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> a = dc.shuffle()
        >>> tuple(a) == tuple(range(5))
        False

        2. streamed data collection is not supported:

        >>> dc = DataCollection([0, 1, 2, 3, 4]).stream()
        >>> _ = dc.shuffle()
        Traceback (most recent call last):
        TypeError: shuffle is not supported for streamed data collection.
        """
        if self.is_stream:
            raise TypeError('shuffle is not supported for streamed data collection.')
        iterable = random.sample(self._iterable, len(self._iterable))
        return self._factory(iterable)

    def __getattr__(self, name):
        """
        Unknown method dispatcher.

        When a unknown method is invoked on a `DataCollection` object,
        the function call will be dispatched to a method resolver.
        By registering function to the resolver, you are able to extend
        `DataCollection`'s API at runtime without modifying its code.

        Examples:

        1. Define two operators:

        >>> from towhee import register
        >>> @register
        ... class myadd:
        ...     def __init__(self, val):
        ...         self.val = val
        ...     def __call__(self, x):
        ...         return x+self.val

        >>> @register
        ... class mymul:
        ...     def __init__(self, val):
        ...         self.val = val
        ...     def __call__(self, x):
        ...         return x*self.val

        2. Register the operators to `DataCollection`'s execution context with `param_scope`:

        >>> dc = DataCollection([1,2,3,4])
        >>> dc.myadd(1).mymul(val=2).to_list() # call registered operator
        [4, 6, 8, 10]
        """

        if name.startswith('_'):
            return super().__getattribute__(name)

        @dynamic_dispatch
        def wrapper(*arg, **kws):
            with param_scope() as hp:
                # pylint: disable=protected-access
                path = hp._name
                index = hp._index
            if self.get_backend() == 'ray':
                return self.ray_resolve({}, path, index, *arg, **kws)
            if self._jit is not None:
                op = self.jit_resolve(path, index, *arg, **kws)
            else:
                op = self.resolve(path, index, *arg, **kws)
            return self.map(op)

        return getattr(wrapper, name)

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
            raise TypeError(
                'indexing is only supported for '
                'data collection created from list or pandas DataFrame.')
        if isinstance(index, int):
            return self._iterable[index]
        return DataCollection(self._iterable[index])

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
            raise TypeError(
                'indexing is only supported for '
                'data collection created from list or pandas DataFrame.')
        self._iterable[index] = value

    def append(self, item: Any) -> 'DataCollection':
        """
        Append item to data collection

        Args:
            item (Any): the item to append

        Returns:
            DataCollection: self

        Examples:

        >>> dc = DataCollection([0, 1, 2])
        >>> dc.append(3).append(4)
        [0, 1, 2, 3, 4]
        """
        if hasattr(self._iterable, 'append'):
            self._iterable.append(item)
            return self
        raise TypeError('appending is only supported for '
                        'data collection created from list.')

    @register_dag
    def __add__(self, other):
        """
        Concat two data collections:

        Examples:

        >>> (DataCollection.range(5) + DataCollection.range(5)).to_list()
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

        >>> (DataCollection.range(5) + DataCollection.range(5) + DataCollection.range(5)).to_list()
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        """
        self.parent_ids.append(other.id)
        other.notify_consumed(self.id)

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

        >>> DataCollection([1, 2, 3]).unstream()
        [1, 2, 3]

        >>> DataCollection([1, 2, 3]).stream() #doctest: +ELLIPSIS
        <list_iterator object at...>
        """
        if isinstance(self._iterable, list):
            return reprlib.repr(self._iterable)
        if hasattr(self._iterable, '__repr__'):
            return repr(self._iterable)
        return super().__repr__()

    @register_dag
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

        def inner():
            for i, x in enumerate(self._iterable):
                if i >= n:
                    break
                yield x

        return self._factory(inner())

    def run(self):
        for _ in self._iterable:
            pass

    def to_list(self):
        return self._iterable if isinstance(self._iterable, list) else list(self._iterable)

    def to_df(self):
        """
        Turn a DataCollection to DataFrame.

        Examples:

        >>> from towhee import DataCollection, Entity
        >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
        >>> dc = DataCollection(e)
        >>> type(dc)
        <class 'towhee.functional.data_collection.DataCollection'>
        >>> type(dc.to_df())
        <class 'towhee.functional.data_collection.DataFrame'>
        """
        return DataFrame(self._iterable)


class DataFrame(DataCollection, DataFrameMixin, ColumnMixin):
    """
    Entity based DataCollection.

    Args:
        iterable (Iterable): input data.

    >>> from towhee import Entity
    >>> DataFrame([Entity(id=a) for a in [1,2,3]])
    [<Entity dict_keys(['id'])>, <Entity dict_keys(['id'])>, <Entity dict_keys(['id'])>]
    """

    def __init__(self, iterable: Iterable = None, **kws) -> None:
        """Initializes a new DataCollection instance.

        Args:
            iterable (Iterable): input data
        """
        if iterable is not None:
            super().__init__(iterable)
            self._mode = self.ModeFlag.ROWBASEDFLAG
        else:
            super().__init__(DataFrame.from_arrow_talbe(**kws))
            self._mode = self.ModeFlag.COLBASEDFLAG


    def _factory(self, iterable, parent_stream=True, mode=None):
        """
        Factory method for DataFrame.

        This factory method has been wrapped into a `param_scope()` which contains parent information.

        Args:
            iterable:
                An iterable object, the data being stored in the DC
            parent_stream:
                Whether to copy the parents format (streamed vs unstreamed)

        Returns:
            DataFrame: DataFrame encapsulating the iterable.
        """

        # pylint: disable=protected-access
        if parent_stream is True:
            if self.is_stream:
                if not isinstance(iterable, Iterator):
                    iterable = iter(iterable)
            else:
                if isinstance(iterable, Iterator):
                    iterable = list(iterable)

        with param_scope() as hp:
            hp().data_collection.parent = self
            df = DataFrame(iterable)
            df._mode = self._mode if mode is None else mode
            return df

    def to_dc(self):
        """
        Turn a DataFrame to DataCollection.

        Examples:

        >>> from towhee import DataFrame, Entity
        >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
        >>> df = DataFrame(e)
        >>> type(df)
        <class 'towhee.functional.data_collection.DataFrame'>
        >>> type(df.to_dc())
        <class 'towhee.functional.data_collection.DataCollection'>
        """
        return DataCollection(self._iterable)

    @property
    def mode(self):
        """
        Return the storage mode of the DataFrame.

        Examples:

        >>> from towhee import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(range(5), range(5))]
        >>> df = DataFrame(e)
        >>> df.mode
        <ModeFlag.ROWBASEDFLAG: 1>
        >>> df = df.to_column()
        >>> df.mode
        <ModeFlag.COLBASEDFLAG: 2>
        """
        return self._mode

    def __iter__(self):
        """
        Define the way of iterating a DataFrame.

        Examples:

        >>> from towhee import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(range(3), range(3))]
        >>> df = DataFrame(e)
        >>> df.to_list()[0]
        <Entity dict_keys(['a', 'b'])>
        >>> df = df.to_column()
        >>> df.to_list()[0]
        <EntityView dict_keys(['a', 'b'])>
        >>> df = DataFrame(e)
        >>> df = df.set_chunksize(2)
        >>> df.to_list()[0]
        <EntityView dict_keys(['a', 'b'])>
        """
        if hasattr(self._iterable, 'iterrows'):
            # for i in self._iterable.iterrows():
            #     yield i[i]
            return (x[1] for x in self._iterable.iterrows())
        if self._mode == self.ModeFlag.ROWBASEDFLAG:
            # for i in self._iterable:
            #     yield i
            return iter(self._iterable)
        if self._mode == self.ModeFlag.COLBASEDFLAG:
            # for i in range(len(self._iterable)):
            #     yield EntityView(i, self._iterable)
            return (EntityView(i, self._iterable) for i in range(len((self._iterable))))
        if self._mode == self.ModeFlag.CHUNKBASEDFLAG:
            # for wtable in self._iterable.chunks():
            #     for ev in wtable:
            #         yield ev
            return (ev for wtable in self._iterable.chunks() for ev in wtable)

    def map(self, *arg):
        if hasattr(arg[0], '__check_init__'):
            arg[0].__check_init__()
        if self._mode == self.ModeFlag.COLBASEDFLAG or self._mode == self.ModeFlag.CHUNKBASEDFLAG:
            return self.cmap(arg[0])
        else:
            return super().map(*arg)
