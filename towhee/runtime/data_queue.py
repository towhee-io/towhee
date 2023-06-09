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
import copy
from enum import Enum, auto
from typing import List, Tuple, Union, Dict, Optional

from collections import deque, namedtuple


class DataQueue:
    """
    Col-based storage.
    """

    def __init__(self, schema_info, max_size=1000, keep_data=False):
        self._max_size = max_size
        self._schema = _Schema(schema_info)
        self._data = []
        self._queue_index = []
        self._scalar_index = []
        self._has_all_scalars = False
        self._readed = False
        for index in range(len(self._schema.col_types)):
            col_type = self._schema.col_types[index]
            if col_type == ColumnType.QUEUE:
                self._data.append(_QueueColumn() if not keep_data else _ListColumn())
                self._queue_index.append(index)
            else:
                self._data.append(_ScalarColumn())
                self._scalar_index.append(index)

        self._sealed = False
        self._size = 0
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def put(self, inputs: Union[Tuple, List]) -> bool:
        assert len(inputs) == self._schema.size()
        with self._not_full:
            if self._sealed:
                return False

            if self._max_size > 0:
                while self.size >= self._max_size:
                    self._not_full.wait()

            for i in range(len(inputs)):
                self._data[i].put(inputs[i])

            self._size = self._get_size()
            if self._size > 0:
                self._not_empty.notify(self._size)
            return True

    def put_dict(self, inputs: Dict) -> bool:
        data = [inputs.get(name, Empty()) for name in self._schema.col_names]
        return self.put(data)

    def batch_put(self, batch_inputs: List[List]) -> bool:
        assert len(batch_inputs) == self._schema.size()
        with self._not_full:
            if self._sealed:
                return False

            if self._max_size > 0:
                while self.size >= self._max_size:
                    self._not_full.wait()

            for col_index in range(self._schema.size()):
                if self._schema.get_col_type(col_index) == ColumnType.SCALAR:
                    self._data[col_index].put(batch_inputs[col_index][0])
                else:
                    for item in batch_inputs[col_index]:
                        self._data[col_index].put(item)

            self._size = self._get_size()
            if self._size > 0:
                self._not_empty.notify(self._size)
            return True

    def batch_put_dict(self, batch_inputs: Dict) -> bool:
        need_put = False
        cols = []
        for name in self._schema.col_names:
            col = batch_inputs.get(name)
            if not col:
                cols.append([Empty()])
            else:
                need_put = True
                cols.append(col)
        if need_put:
            return self.batch_put(cols)
        return True

    def get(self) -> Optional[List]:
        with self._not_empty:
            while self._size <= 0 and not self._sealed:
                self._not_empty.wait()

            if self._size <= 0:
                return None

            ret = []
            for col in self._data:
                ret.append(col.get())
            self._size -= 1
            self._readed = True
            self._not_full.notify()
            return ret

    def get_dict(self, cols: List[str] = None) -> Optional[Dict]:
        data = self.get()
        if data is None:
            return None

        ret = {}
        names = self._schema.col_names
        if cols is None:
            for i, name in enumerate(names):
                ret[name] = data[i]
        else:
            for i, name in enumerate(names):
                if name in cols:
                    ret[name] = data[i]
        return ret

    def to_list(self, kv_format=False):
        if not self.sealed:
            raise RuntimeError('The queue is not sealed')
        if not kv_format:
            return [self.get() for _ in range(self.size)]
        return [self.get_dict() for _ in range(self.size)]

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, max_size):
        with self._not_full:
            need_notify = self._max_size > max_size or max_size == 0
            self._max_size = max_size
            if need_notify:
                self._not_full.notify_all()

    @property
    def size(self) -> int:
        return self._size

    @property
    def col_size(self) -> int:
        return self._schema.size()

    def clear_and_seal(self):
        with self._lock:
            if self._sealed:
                return
            self._size = 0
            self._sealed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def seal(self):
        with self._lock:
            if self._sealed:
                return

            self._sealed = True
            if self._queue_index or not self._has_all_scalars:
                self._size = self._get_size()
            self._not_empty.notify_all()
            self._not_full.notify_all()

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def schema(self) -> List[str]:
        """
        Return the schema of the DataQueue.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            True
            >>> dq.schema
            ['a', 'b']
        """
        return self._schema.col_names

    @property
    def type_schema(self) -> List[str]:
        """
        Return the type of queues in the DataQueue.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            True
            >>> dq.type_schema
            [<ColumnType.SCALAR: 2>, <ColumnType.QUEUE: 1>]
        """
        return self._schema.col_types

    def _get_size(self):
        if not self._sealed and not self._has_all_scalars:
            for index in self._scalar_index:
                if not self._data[index].has_data():
                    return 0
            self._has_all_scalars = True

        if self._queue_index:
            que_size = [self._data[index].size() for index in self._queue_index]
            if not self._sealed:
                return min(que_size)

            size = max(que_size)
            if size > 0 or self._readed:
                return size

        for index in self._scalar_index:
            if self._data[index].has_data():
                return 1
        return 0

    def col_type(self, col_name):
        return self.type_schema[self.schema.index(col_name)]

    def reset_size(self):
        """
        For debug, read data repeatedly.
        """
        for col in self._data:
            if isinstance(col, _ListColumn):
                col.reset_size()
        self._size = self._get_size()


class ColumnType(Enum):
    """
    ColumnType
    """
    QUEUE = auto()
    SCALAR = auto()


_ColumnInfo = namedtuple('_ColumnInfo', ['name', 'col_type'])


class _Schema:
    """
    schema_info.
    """

    def __init__(self, schema_info: List[Tuple]):
        self._cols = []
        for col in schema_info:
            self._cols.append(_ColumnInfo(*col))
        self._size = len(schema_info)

    def size(self):
        return self._size

    @property
    def col_names(self):
        return [col.name for col in self._cols]

    @property
    def col_types(self):
        return [col.col_type for col in self._cols]

    def get_col_name(self, index):
        assert index < self._size
        return self._cols[index].name

    def get_col_type(self, index):
        assert index < self._size
        return self._cols[index].col_type


class Empty:
    """
    Empty data.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            with cls._lock:
                cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __str__(self):
        return 'Empty()'

    def __repr__(self):
        return 'Empty()'


class _QueueColumn:
    """
    Queue column.
    """

    def __init__(self):
        self._q = deque()

    def get(self):
        if len(self._q) == 0:
            return Empty()
        return self._q.popleft()

    def put(self, data) -> bool:
        if data is Empty():
            return
        self._q.append(data)

    def size(self):
        return len(self._q)


class _ListColumn:
    """
    List column, for debug.
    """

    def __init__(self):
        self._q = []
        self._index = 0

    def get(self):
        if len(self._q) == self._index:
            return Empty()
        self._index += 1
        return copy.deepcopy(self._q[self._index - 1])

    def put(self, data) -> bool:
        if data is Empty():
            return
        self._q.append(data)

    def size(self):
        return len(self._q) - self._index

    def reset_size(self):
        self._index = 0


class _ScalarColumn:
    """
    Scalar column
    """

    def __init__(self):
        self._data = Empty()

    def put(self, data):
        if data is Empty():
            return
        self._data = data

    def get(self):
        return self._data

    def has_data(self):
        return self._data is not Empty()
