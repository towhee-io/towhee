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
from towhee.functional.mixins.dag import register_dag
from typing import Iterable

from towhee.functional.entity import Entity


class DataProcessingMixin:
    """
    Mixin for processing data.
    """
    @register_dag
    def select_from(self, other):
        """
        Select data from dc with list(self).

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
        """
        Combine two data collections.

        Args:
            *others (DataCollection): other data collections;

        Returns:
            DataCollection: data collection with zipped values;

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
        """
        Get the first n lines of a DataCollection.

        Args:
            n (`int`):
                The number of lines to print. Default value is 5.

        Examples:

        >>> from towhee import DataCollection
        >>> DataCollection.range(10).head(3).to_list()
        [0, 1, 2]
        """
        def inner():
            for i, x in enumerate(self._iterable):
                if i >= n:
                    break
                yield x

        return self._factory(inner())

    @register_dag
    def sample(self, ratio=1.0) -> 'DataCollection':
        """
        Sample the data collection.

        Args:
            ratio (float): sample ratio;

        Returns:
            DataCollection: sampled data collection;

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
    def batch(self, size, drop_tail=False, raw=True):
        """
        Create small batches from data collections.

        Args:
            size (`int`):
                Window size;
            drop_tail (`bool`):
                Drop tailing windows that not full, defaults to False;
            raw (`bool`):
                Whether to return raw data instead of DataCollection, defaults to True

        Returns:
            DataCollection of batched windows or batch raw data

        Examples:

        >>> from towhee import DataCollection
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
            size (`int`):
                Wndow size.
            drop_head (`bool`):
                Drop headding windows that not full.
            drop_tail (`bool`):
                Drop tailing windows that not full.

        Returns:
            DataCollection: data collection of rolling windows;

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

        >>> from towhee import DataCollection
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

        >>> from towhee import DataCollection
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
