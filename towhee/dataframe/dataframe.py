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

import threading
from typing import Callable, List, Tuple
import weakref

from towhee.dataframe.variable import Variable


class DataFrame:
    """A dataframe is a collection of immutable, potentially heterogeneous blogs of
    data.
    """

    def __init__(self, name: str, data: List[Tuple[Variable]] = None):
        """DataFrame constructor.

        Args:
            name:
                Name of the dataframe; `DataFrame` names should be the same as its
                representation.
            data:
                A list of data tuples - in all instances, the number of elements per
                tuple should be identical throughout the entire lifetime of the
                `Dataframe`. These tuples can be interpreted as being direct outputs
                into downstream operators.
        """
        self._name = name
        self._data = data if data is not None else []
        self._registered_iter = []
        self._start_index = 0
        self._total = 0
        self._sealed = False

        self._lock = threading.Lock()
        self._registered_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    def put(self, item: Tuple[Variable]) -> None:
        assert not self._sealed, 'DataFrame %s is already sealed, can not put data' % self._name
        with self._lock:
            self._data.append(item)
            self._total += 1

    def get(self, start: int, count: int, force_size: bool = False) -> Tuple[bool, List[Tuple[Variable]]]:
        """
        Get [start: start + count) items of dataframe
        `force_size` is true, return data size must equal `count`. If there is
        not enough data, return empty list.
        `force_size` is false, return no more than `count` data

        Returns:
            (is_sealed, data)
        """
        # TODO (junjie.jiang)
        # 1. call gc function
        # 2. put and seal will be called by the same operator
        # 3. seal will always be called after put
        # 4. iterators never ends until they meet df._sealed == True
        with self._lock:
            if start < self._start_index:
                raise IndexError(
                    'Can not read from {start}, dataframe {name} start index is {cur_start}',
                    start=start,
                    name=self.name,
                    cur_start=self._start_index
                )

            if force_size and not self._sealed and self._total - start < count:
                return False, []

            real_count = min(self._total - start, count)
            real_start_index = start - self._start_index
            return self._sealed and real_start_index + real_count >= self._total, \
                self._data[real_start_index: real_start_index + real_count]

    @property
    def size(self):
        return self._total

    def seal(self):
        with self._lock:
            self._sealed = True

    def is_sealed(self) -> bool:
        return self._sealed

    # TODO (junjie.jiangjjj)
    def _gc(self) -> None:
        """
        Delete the data which all registered iter has read
        """
        return None

    def map_iter(self):
        it = MapIterator(self)
        with self._registered_lock:
            self._registered_iter.append(it)
        return it


class DataFrameIterator:
    """Base iterator implementation. All iterators should be subclasses of
    `DataFrameIterator`.

    Args:
        df: The dataframe to iterate over.
    """

    def __init__(self, df: DataFrame):
        self._df_ref = weakref.ref(df)
        self._cur_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        # The base iterator is purposely defined to have exatly 0 elements.
        raise StopIteration

    @property
    def accessible_size(self) -> int:
        """
        Return current accessible data size
        """
        return self._df_ref().size - self._cur_index


class MapIterator(DataFrameIterator):
    """Iterator implementation that traverses the dataframe line-by-line.

    Args:
        df: The dataframe to iterate over.
    """

    def __init__(self, df: DataFrame):
        super().__init__(df)
        self._lock = threading.Lock()

    def __next__(self) -> List[Tuple]:
        with self._lock:
            sealed, data = self._df_ref().get(self._cur_index, 1)
            if sealed and len(data) == 0:
                raise StopIteration
            self._cur_index += len(data)
        return data


class BatchIterator:
    """Iterator implementation that traverses the dataframe multiple rows at a time.

    Args:
        df:
            The dataframe to iterate over.
        size:
            batch size. If size is None, then the whole variable set will be batched
            together.
    """

    def __init__(self, df: DataFrame, size: int = None):
        super().__init__(df)
        self._size = size

    def __next__(self):
        raise NotImplementedError


class GroupIterator:
    """Iterator implementation that traverses the dataframe based on a custom group-by
    function.

    Args:
        df:
            The dataframe to iterate over.
        func:
            The function used to return grouped data within the dataframe.
    """

    def __init__(self, df: DataFrame, func: Callable[[], None]):
        super().__init__(df)
        self._func = func

    def __next__(self):
        raise NotImplementedError


class RepeatIterator:
    """Iterator implementation that repeatedly accesses the first line of the dataframe.

        Args:
            df:
                The dataframe to iterate over.
            n:
                the repeat times. If `None`, the iterator will loop continuously.
    """

    def __init__(self, df: DataFrame, n: int = None):
        # self._n = n
        # self._i = 0
        raise NotImplementedError

    def __next__(self):
        if self._i >= self._n:
            raise StopIteration
        return self._data[0]
