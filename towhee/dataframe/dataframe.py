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
from typing import Callable, List
import weakref

from towhee.array import Array


class DataFrame:
    """A dataframe is a collection of immutable, potentially heterogeneous arrays of
       data.

    Args:
        name: (`str`)
            Name of the `DataFrame`. A `DataFrame`'s name and its coupled 
            `DataFrameRepr`'s name should be consistent.
        data: (`list` of `Array`)
            The data of the `DataFrame`. Each `Array` in the list will form a column
            in the `DataFrame`.
    """
    def __init__(self, name: str, data: List[Array] = None):

        self._name = name
        self._columns = data if data is not None else []
        self._columns_dict = {(col.name, col) for col in data}
        self._iters = []
        self._sealed = False
        self._size = 0

        self._lock = threading.Lock()

    def __getitem__(self, key):
        """
        Args: 
            key: (`int` or `str`)
                The column index or column name
        """
        if isinstance(key, int):
            return self._columns[key]
        elif isinstance(key, str):
            return self._columns_dict[key]
        else:
            raise IndexError("only integers or strings are invalid indices")

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> List:
        return self._columns

    @property
    def size(self) -> int:
        return self._size

    def put(self, row: List):
        """Append one row to the end of this `DataFrame`
        """
        with self._lock:
            assert not self._sealed, 'DataFrame %s is already sealed, can not put data' % self._name

            for i, v in row:
                self._columns[i].put(v)
            self._size += 1

    def append(self, data):
        """Append a dataframe-like data to the end of this `DataFrame`.
        Args:
            data: (`list` of `Array`, or `DataFrame`)
                The data to be appended.
        """
        if isinstance(data, DataFrame):
            data = data._columns
        size = data[0].size

        for v in data.items():
            if v.size != size:
                raise ValueError(
                    'Each column of the appended data should be equally sized.'
                )

        with self._lock:
            assert not self._sealed, 'DataFrame %s is already sealed, can not append data' % self._name

            for i, v in data.items():
                self._columns[i].append(v)
            self._size += size

    def is_empty(self):
        return not self._size

    def seal(self):
        with self._lock:
            self._sealed = True

    def is_sealed(self) -> bool:
        return self._sealed

    def map_iter(self):
        it = MapIterator(self, self._wcond)
        self._iters.append(it)
        return it


class DataFrameIterator:
    """Base iterator implementation. All `DataFrame` iterators should be derived from
    `DataFrameIterator`.

    Args:
        df: (`DataFrame`)
            The dataframe to iterate over.
    """
    def __init__(self, df: DataFrame):
        self._df_ref = weakref.ref(df)

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class MapIterator(DataFrameIterator):
    """Iterator implementation that traverses the dataframe line-by-line.

    Args:
        df: (`DataFrame`)
            The dataframe to iterate over.
    """
    def __init__(self, df: DataFrame):
        super().__init__(df)
        self._lock = threading.Lock()
        self._cur_index = 0

    def __next__(self) -> List:
        with self._lock:
            # if there is a ready row in df
            if self._has_next_row():
                return self._get_next_row()
            # next row is not ready yet
            if not self._df_ref.is_sealed:
                return None
            # df is sealed, see if there are rows left
            elif self._has_next_row():
                return self._get_next_row()
            # the iteration ends
            else:
                raise StopIteration

    def _has_next_row(self) -> bool:
        return self._cur_index < self._df_ref.size

    def _get_next_row(self):
        i = self._cur_index
        # next row is ready
        data = [col[i] for col in self._df_ref.columns]
        self._cur_index += 1
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
