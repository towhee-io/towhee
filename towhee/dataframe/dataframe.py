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
from typing import Any, Callable, List, Tuple, Dict
import weakref

from towhee.dataframe.variable import Variable


class DataFrame:
    """A dataframe is a collection of immutable, potentially heterogeneous blogs of
    data.
    """

    def __init__(self, name: str = None, cols=None, data: List[Tuple[Variable]] = None):
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
        # TODO (junjie.jiangjjj) define col struct
        self._cols = cols
        self._data = data if data else []

        # `_start_idx` corresponds to the actual index for the current 0th element in
        # this `DataFrame`.
        self._start_idx = 0
        self._total = len(self._data)

        # A list of `DataFrameIterator` instances registered to this `DataFrame`.
        self._iters = []

        # Any changes made to `DataFrame` state must first acquire this lock.
        self._lock = threading.Lock()

        # This flag and condition variable track whether or not this `DataFrame` can
        # still be used. Sealed `DataFrames` are disabled.
        self._sealed = False
        self._seal_cv = threading.Condition(self._lock)

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> List[Tuple[Variable]]:
        return self._data

    @property
    def size(self) -> int:
        return self._total

    @property
    def sealed(self) -> bool:
        return self._sealed

    def put(self, item: Tuple[Variable]) -> None:
        assert not self._sealed, f'DataFrame {self._name} is already sealed, can not put data'
        assert isinstance(item, tuple), 'Dataframe needs tuple, not %s' % (type(item))
        with self._lock:
            self._data.append(item)
            self._total += 1

    def put_dict(self, data: Dict[str, Any]):
        datalist = [None] * len(self._cols)
        for k, v in data.items():
            datalist[self._cols[k]['index']] = Variable(
                self._cols[k]['type'], v)
        self.put(tuple(datalist))

    def merge(self, df):
        assert not self._sealed, f'DataFrame {self._name} is already sealed, can not put data'
        # TODO: Check `df` compatibility with `self`.
        with self._lock:
            self._data += df.data
            self._total += len(df.data)

    def get(self, start: int, count: int) -> List[Tuple[Variable]]:
        """Get [start: start + count) elements within the `DataFrame`. If `wait` is
        `True`, this function will block until either the `DataFrame` is sealed or there
        are enough elements.

        Args:
            start: (`int`)
                The start index within the `DataFrame`.
            count: (`int`)
                Number of elements to acquire.

        Returns:
            data: (`List[Tuple[Variable]]`)
        """

        # TODO (junjie.jiang)
        # 1. call gc function
        # 2. put and seal will be called by the same operator
        # 3. seal will always be called after put
        # 4. iterators never ends until they meet df._sealed == True
        with self._lock:
            if start < self._start_idx:
                raise IndexError(
                    'Can not read from {start}, dataframe {name} start index is {cur_start}',
                    start=start,
                    name=self._name,
                    cur_start=self._start_idx
                )

            # If there are enough elements within the `DataFrame`, gather the
            # outputs and return them.
            if self._total - start >= count:
                idx0 = start - self._start_idx
                return self._data[idx0:idx0+count]

            # If the `DataFrame` is already sealed, return only the remaining data.
            if self._sealed:
                if self._total <= start:
                    return None
                return self._data[start:]

            return None

    def seal(self):
        with self._seal_cv:
            self._sealed = True
            self._seal_cv.notify_all()

    def wait_sealed(self):
        with self._seal_cv:
            if not self._sealed:
                self._seal_cv.wait()

    def clear(self):
        with self._lock:
            self._data = []
            self._start_idx = 0
            self._total = 0
            self._sealed = False

    def _gc(self) -> None:
        # TODO (junjie.jiangjjj)
        """
        Delete the data which all registered iter has read
        """
        return

    def map_iter(self):
        it = MapIterator(self)
        with self._lock:
            self._iters.append(it)
        return it

    def __str__(self) -> str:
        return 'DataFrame [%s] with [%s] datas, seal: [%s]' % (self._name, self._size, self._sealed)


class DataFrameIterator:
    """Base iterator implementation. All iterators should be subclasses of
    `DataFrameIterator`.

    Args:
        df: The dataframe to iterate over.
    """

    def __init__(self, df: DataFrame):
        self._df_ref = weakref.ref(df)
        self._cur_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> List[Tuple]:
        # The base iterator is purposely defined to have exatly 0 elements.
        raise StopIteration

    @property
    def current_index(self) -> int:
        """Returns the current index.
        """
        return self._cur_idx

    @property
    def accessible_size(self) -> int:
        """Returns current accessible data size.
        """
        return self._df_ref().size - self._cur_idx


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
            data = self._df_ref().get(self._cur_idx, 1)
            if self._df_ref().sealed and not data:
                raise StopIteration
            if data:
                self._cur_idx += 1
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
