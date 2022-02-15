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
from typing import Any, List, Tuple, Dict
import weakref

from towhee.utils.atomic_count import AtomicCount
from towhee.dataframe.variable import Variable
from towhee.types._frame import FRAME, _Frame
from towhee.dataframe._schema import _Schema


class DataFrame:
    """
    A dataframe is a collection of immutable, potentially heterogeneous blogs of data.
    """

    def __init__(self, name: str = None, cols=None, data: List[Tuple[Variable]] = None):
        """DataFrame constructor.

        Args:
            name (`str`):
                Name of the dataframe; `DataFrame` names should be the same as its
                representation.
            clos (`List[Tuple(str, str)]`):
                Dataframe cols.
            data (`List[Tuple[Variable]]`):
                A list of data tuples - in all instances, the number of elements per
                tuple should be identical throughout the entire lifetime of the
                `Dataframe`. These tuples can be interpreted as being direct outputs
                into downstream operators.
        """
        self._name = name
        self._schema = _Schema()
        self._add_cols(cols)
        self._data = data if data else []

        # `_start_idx` corresponds to the actual index for the current 0th element in
        # this `DataFrame`.
        self._start_idx = 0
        self._total = len(self._data)

        # A list of `DataFrameIterator` instances registered to this `DataFrame`.
        self._iters = []

        # Sealed `DataFrames` are disabled, while changes made to `DataFrame` state must
        # first acquire this lock. This flag and condition variable track whether or not
        # this `DataFrame` can still be used.
        self._sealed = False
        self._lock = threading.Lock()
        self._seal_cv = threading.Condition(self._lock)
        self._accessible_cv = threading.Condition(self._lock)

    def _add_cols(self, cols):
        if cols is not None:
            for col in cols:
                self._schema.add_col(*col)
        self._schema.add_col(FRAME, '_Frame')
        self._schema.seal()

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
    def current_size(self):
        return max([self._total - self._start_idx, 0])

    @property
    def sealed(self) -> bool:
        """
        Sealed `DataFrame`s are disabled.
        """
        return self._sealed

    def _set_frame(self, item):
        if len(item) != 0 and isinstance(item[-1], Variable) and isinstance(item[-1].value, _Frame):
            item[-1].value.row_id = self._total
        else:
            f = _Frame(row_id=self._total)
            item = list(item)
            item.append(Variable(FRAME, f))
            item = tuple(item)
        return item

    def put(self, item: Tuple[Variable]) -> None:
        assert not self._sealed, f'DataFrame {self._name} is already sealed, can not put data'
        assert isinstance(item, tuple), 'Dataframe needs tuple, not %s' % (type(item))
        with self._lock:
            item = self._set_frame(item)
            if item[-1].value.empty:
                return
            self._data.append(item)
            self._total += 1
            self._accessible_cv.notify()

    def put_dict(self, data: Dict[str, Any]):
        datalist = [None] * self._schema.col_count
        for k, v in data.items():
            col_index = self._schema.col_index(k)
            datalist[col_index] = Variable(
                self._schema.col_type(col_index), v)
        self.put(tuple(datalist))

    def get(self, start: int, count: int, block: bool = False) -> List[Tuple[Variable]]:
        """
        Get [start: start + count) elements within the `DataFrame`.

        If there are enough elements within this `DataFrame`, this function will gather
        the elements and return them. If this `DataFrame` is already sealed and fewer
        than `count` elements are available, this function will return all the remaining
        elements. In all other cases, this function will return `None`.

        Args:
            start: (`int`)
                The start index within the `DataFrame`.
            count: (`int`)
                Number of elements to acquire.

        Returns:
            data: (`List[Tuple[Variable]]`)
                At most `count` elements acquired from the `DataFrame`, or `None` if
                not enough elements are available.
        """

        with self._accessible_cv:
            if start < self._start_idx:
                raise IndexError(
                    'Can not read from {start}, dataframe {name} start index is {cur_start}',
                    start=start,
                    name=self._name,
                    cur_start=self._start_idx
                )

            if block and not self._accessible(start, count):
                self._accessible_cv.wait()

            if self._total - start >= count:
                idx0 = start - self._start_idx
                return self._data[idx0:idx0+count]

            # If the `DataFrame` is already sealed, return only the remaining data.
            if self._sealed:
                if self._total <= start:
                    return None
                return self._data[start - self._start_idx:]
            return None

    def notify_block_readers(self):
        with self._accessible_cv:
            self._accessible_cv.notify_all()

    def _accessible(self, start: int, count: int) -> bool:
        if self._sealed:
            return True
        if self._total - start >= count:
            return True
        return False

    def seal(self):
        with self._seal_cv:
            self._sealed = True
            self._seal_cv.notify_all()
            self._accessible_cv.notify_all()

    def wait_sealed(self):
        with self._seal_cv:
            while not self._sealed:
                self._seal_cv.wait()

    def clear(self):
        with self._lock:
            self._data = []
            self._start_idx = 0
            self._total = 0
            self._sealed = False

    def gc(self) -> None:
        """
        Delete the data which all registered iter has read
        """
        with self._lock:
            if len(self._iters) == 0:
                return

            min_index = min([it.current_index for it in self._iters])
            if min_index > self._start_idx:
                self._data = self._data[min_index - self._start_idx:]
                self._start_idx = min_index

    def map_iter(self, block: bool = False):
        it = MapIterator(self, block)
        with self._lock:
            self._iters.append(it)
        return it

    def batch_iter(self, size: int, step: int, block: bool = False):
        it = BatchIterator(self, size, step, block)
        with self._lock:
            self._iters.append(it)
        return it

    def __str__(self) -> str:
        return 'DataFrame [%s] with [%s] datas, seal: [%s]' % (self._name, self.size, self._sealed)


class DataFrameIterator:
    """Base iterator implementation. All iterators should be subclasses of
    `DataFrameIterator`.

    Args:
        df: The dataframe to iterate over.
    """

    def __init__(self, df: DataFrame):
        self._df_ref = weakref.ref(df)
        self._cur_idx = AtomicCount(0)
        self._df_name = df.name

    def __iter__(self):
        return self

    def __next__(self) -> List[Tuple]:
        # The base iterator is purposely defined to have exatly 0 elements.
        raise StopIteration

    @property
    def df_name(self) -> int:
        """Returns the df name.
        """
        return self._df_name

    @property
    def current_index(self) -> int:
        """Returns the current index.
        """
        return self._cur_idx.count

    @property
    def accessible_size(self) -> int:
        """Returns current accessible data size.
        """
        return self._df_ref().size - self._cur_idx.count

    def notify(self):
        df = self._df_ref()
        if df is not None:
            self._df_ref().notify_block_readers()


class MapIterator(DataFrameIterator):
    """Iterator implementation that traverses the dataframe line-by-line.

    Args:
        df: The dataframe to iterate over.
    """

    def __init__(self, df: DataFrame, block: bool = False):
        super().__init__(df)
        self._block = block

    def __next__(self) -> List[Tuple]:
        data = self._df_ref().get(self._cur_idx.count, 1, self._block)
        if self._df_ref().sealed and not data:
            raise StopIteration
        if data is not None:
            self._cur_idx += 1
            return data[0]
        return None


class BatchIterator(DataFrameIterator):
    """Iterator implementation that traverses the dataframe multiple rows at a time.

    Args:
        df:
            The dataframe to iterate over.
        size:
            batch size. If size is None, then the whole variable set will be batched
            together.
    """

    def __init__(self, df: DataFrame, size: int, step: int, block=False):
        super().__init__(df)
        self._size = size
        self._step = step
        self._block = block

    def __next__(self):
        data = None
        while data is None:
            df = self._df_ref()
            data = df.get(self._cur_idx.count, self._size, self._block)
            if not data and self._df_ref().sealed and self._cur_idx.count >= df.size:
                raise StopIteration

            if data is not None and len(data) > 0:
                self._cur_idx += self._step
                return data

            return None
