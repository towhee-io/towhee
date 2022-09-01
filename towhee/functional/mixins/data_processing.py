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
import random
from typing import Iterable

from towhee.functional.entity import Entity
from towhee.functional.mixins.dag import register_dag
from towhee.hparam.hyperparameter import dynamic_dispatch, param_scope

class DataProcessingMixin:
    """
    Mixin for processing data.
    """

    @classmethod
    def combine(cls, *datacollections):
        """Combine dataframes to be able to access schemas from seperate DF chains.

        Args:
            datacollections (DataFrame): DataFrames to combine.

        Examples:
            >>> import towhee
            >>> a = towhee.range['a'](1,5)
            >>> b = towhee.range['b'](5,10)
            >>> c = towhee.range['c'](10, 15)
            >>> z = towhee.DataFrame.combine(a, b, c)
            >>> z.as_raw().to_list()
            [(1, 5, 10), (2, 6, 11), (3, 7, 12), (4, 8, 13)]
        """
        def inner():
            for dicts in zip(*datacollections):
                dicts[0].combine(*dicts[1:])
                yield dicts[0]
        return cls(inner())

    @register_dag
    def select_from(self, other):
        """Select data from dc with list(self).

        Args:
            other (DataCollection): DataCollection to select from.

        Examples:

            >>> from towhee import DataCollection
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
    def zip(self, *others) -> 'DataCollection':
        """Combine multiple data collections.

        Args:
            *others (DataCollection): The other data collections.

        Returns:
            DataCollection: Data collection with zipped values.

        Examples:

            >>> from towhee import DataCollection
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
    def head(self, n: int = 5):
        """Return the first `n` values of a DataCollection.

        Args:
            n (int, optional): The amount to select, defaults to 5.

        Returns:
            DataCollection: DataCollection with the selected values.
        """
        def inner():
            for i, x in enumerate(self._iterable):
                if i >= n:
                    break
                yield x

        return self._factory(inner())

    @register_dag
    def sample(self, ratio=1.0) -> 'DataCollection':
        """Sample the data collection.

        Args:
            ratio (float): sample ratio.

        Returns:
            DataCollection: Sampled data collection.

        Examples:

            >>> from towhee import DataCollection
            >>> dc = DataCollection(range(10000))
            >>> result = dc.sample(0.1)
            >>> ratio = len(result.to_list()) / 10000.
            >>> 0.09 < ratio < 0.11
            True
        """
        return self._factory(filter(lambda _: random.random() < ratio, self))

    @register_dag
    def batch(self, size, drop_tail=False):
        """Create batches from the DataCollection.

        Args:
            size (int): Window size.
            drop_tail (`bool`): Drop trailing window that is not full, defaults to False.


        Returns:
            DataCollection: Batched DataCollection.

        Examples:

            >>> from towhee import DataCollection
            >>> dc = DataCollection(range(10))
            >>> [list(batch) for batch in dc.batch(2)]
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
            [[<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>], [<Entity dict_keys(['a', 'b'])>]]
        """
        def inner():
            buff = []
            count = 0
            for ele in self._iterable:
                buff.append(ele)
                count += 1

                if count == size:
                    yield buff
                    buff = []
                    count = 0

            if not drop_tail and count > 0:
                yield buff

        return self._factory(inner())

    @register_dag
    def rolling(self, size: int, step: int=1, drop_head=True, drop_tail=True):
        """Create rolling windows from DataCollection.

        Args:
            size (int): Window size.
            drop_head (bool): Drop head windows that are not full.
            drop_tail (bool): Drop trailing windows that are not full.

        Returns:
            DataCollection: DataCollection of rolling windows.

        Examples:

            >>> from towhee import DataCollection
            >>> dc = DataCollection(range(5))
            >>> [list(batch) for batch in dc.rolling(3)]
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]]

            >>> dc = DataCollection(range(5))
            >>> [list(batch) for batch in dc.rolling(3, drop_head=False)]
            [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]

            >>> dc = DataCollection(range(5))
            >>> [list(batch) for batch in dc.rolling(3, drop_tail=False)]
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4], [4]]

            >>> from towhee import DataCollection
            >>> dc = DataCollection(range(5))
            >>> dc.rolling(2, 2, drop_head=False, drop_tail=False)
            [[0], [0, 1], [2, 3], [4]]

            >>> from towhee import DataCollection
            >>> dc = DataCollection(range(5))
            >>> dc.rolling(2, 4, drop_head=False, drop_tail=False)
            [[0], [0, 1], [4]]
        """
        def inner():
            buff = []
            gap = 0
            head_flag = True
            for ele in self._iterable:
                if gap:
                    gap -= 1
                    continue

                buff.append(ele)

                if not drop_head and head_flag or len(buff) == size:
                    yield buff.copy()

                if len(buff) == size:
                    head_flag = False
                    buff = buff[step:]
                    gap = step - size if step > size else 0

            while not drop_tail and buff:
                yield buff
                buff = buff[step:]

        return self._factory(inner())

    @property
    # @register_dag
    def flatten(self) -> 'DataCollection':
        """Flatten nested data within DataCollection.

        Returns:
            DataCollection: Flattened DataCollection.

        Examples:

            >>> from towhee import DataCollection, Entity
            >>> dc = DataCollection(range(10))
            >>> nested_dc = dc.batch(2)
            >>> nested_dc.flatten().to_list()
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            >>> g = (i for i in range(3))
            >>> e = Entity(a=1, b=2, c=g)
            >>> dc = DataCollection([e]).flatten['c']()
            >>> [str(i) for i in dc]
            ["{'a': 1, 'b': 2, 'c': 0}", "{'a': 1, 'b': 2, 'c': 1}", "{'a': 1, 'b': 2, 'c': 2}"]
        """
        @dynamic_dispatch
        def flattener():

            def inner():
                for ele in self._iterable:
                    #pylint: disable=protected-access
                    index = param_scope()._index
                    # With schema
                    if isinstance(ele, Entity):
                        if not index:
                            raise IndexError('Please specify the column to flatten.')
                        else:
                            new_ele = ele.__dict__.copy()
                            for nested_ele in getattr(ele, index):
                                new_ele[index] = nested_ele
                                yield Entity(**new_ele)
                    # Without schema
                    elif isinstance(ele, Iterable):
                        for nested_ele in iter(ele):
                            yield nested_ele
                    else:
                        yield ele

            return self._factory(inner())

        return flattener

    @register_dag
    def shuffle(self) -> 'DataCollection':
        """Shuffle an unstreamed data collection in place.

        Returns:
            DataCollection: Shuffled data collection.

        Examples:

            1. Shuffle:
            >>> from towhee import DataCollection
            >>> dc = DataCollection([0, 1, 2, 3, 4])
            >>> a = dc.shuffle()
            >>> tuple(a) == tuple(range(5))
            False

            2. Streamed data collection is not supported:
            >>> dc = DataCollection([0, 1, 2, 3, 4]).stream()
            >>> _ = dc.shuffle()
            Traceback (most recent call last):
            TypeError: shuffle is not supported for streamed data collection.
        """
        if self.is_stream:
            raise TypeError('shuffle is not supported for streamed data collection.')
        iterable = random.sample(self._iterable, len(self._iterable))
        return self._factory(iterable)
