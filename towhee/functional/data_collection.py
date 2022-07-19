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
from typing import Iterable, Iterator, Callable
import reprlib

from towhee.functional.mixins.dag import register_dag
from towhee.hparam import param_scope, dynamic_dispatch
from towhee.functional.option import Option, Some
from towhee.functional.entity import EntityView
from towhee.functional.mixins import DCMixins
from towhee.functional.mixins.dataframe import DataFrameMixin
from towhee.functional.mixins.column import ColumnMixin
from towhee.functional.entity import Entity

# https://stackoverflow.com/questions/1697501/staticmethod-with-property
# decorator for when you want to do:
# @staticmethod
# @property
class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()

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

    # Generation Related Function
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
    
    @classproperty
    def range_schema(self):
        """
        Generate data collection with ranged numbers.

        Examples:

        >>> DataCollection.range_schema['a'](5).to_list_schema['a']()
        [0, 1, 2, 3, 4]
        """
        @dynamic_dispatch
        def range_function(*arg, **kws):
            index = param_scope()._index
            x = DataCollection(range(*arg, **kws)).map(lambda x: Entity(**{index: x}))
            return x
        return range_function

    @property
    def to_list_schema(self):
        """
        Convert entitis into list.

        Examples:

        1. turn specific dataframe indexes into list:

        >>> from towhee import DataFrame
        >>> (
        ...     DataFrame([(1, 2), (2, 3)])
        ...         .as_entity(schema=['a', 'b'])
        ...         .to_list_schema['a']()
        ... )
        [1, 2]

        2. turn entire dataframe indexes into list:

        >>> from towhee import DataFrame
        >>> (
        ...     DataFrame([(1, 2), (2, 3)])
        ...         .as_entity(schema=['a', 'b'])
        ...         .to_list_schema()
        ... )
        [(1, 2), (2, 3)]

        """
        @dynamic_dispatch
        def to_list_function():
            index = param_scope()._index
            if isinstance(index, str):
                index = (index, )
            def inner(entity: Entity):
                if index is not None and len(index) == 1:
                    return getattr(entity, index[0])
                elif index is not None and index:
                    return tuple(getattr(entity, col) for col in index)
                else:
                    return tuple(getattr(entity, name) for name in entity.__dict__)
            res = self.map(inner)
            return list(res._iterable)
        return to_list_function

    def to_list(self):
        return self._iterable if isinstance(self._iterable, list) else list(self._iterable)

    # Execution Related Function
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

    def run(self):
        """
        Consume iterables in stream mode.
        """
        for _ in self._iterable:
            pass

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
            return (x[1] for x in self._iterable.iterrows())
        if self._mode == self.ModeFlag.ROWBASEDFLAG:
            return iter(self._iterable)
        if self._mode == self.ModeFlag.COLBASEDFLAG:
            return (EntityView(i, self._iterable) for i in range(len((self._iterable))))
        if self._mode == self.ModeFlag.CHUNKBASEDFLAG:
            return (ev for wtable in self._iterable.chunks() for ev in wtable)

    def map(self, *arg):
        if hasattr(arg[0], '__check_init__'):
            arg[0].__check_init__()
        if self._mode == self.ModeFlag.COLBASEDFLAG or self._mode == self.ModeFlag.CHUNKBASEDFLAG:
            return self.cmap(arg[0])
        else:
            return super().map(*arg)

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=False)