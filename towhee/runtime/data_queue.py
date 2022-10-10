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

from typing import List, Tuple, Union, Dict, Optional
import threading
import copy
from enum import Enum, auto
from typing import List, Tuple

from collections import deque, namedtuple


class DataQueue:
    '''
    Col-based storage.
    '''

    def __init__(self, schema_info, max_size=0):
        self._max_size = max_size
        self._schema = schema_info if isinstance(schema_info, _Schema) else _Schema(schema_info)
        self._data = []
        for col_type in self._schema.col_types:
            if col_type == ColumnType.QUEUE:
                self._data.append(_QueueColumn())
            else:
                self._data.append(_ScalarColumn())

        self._sealed = False
        self._size = 0
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
    
    def __repr__(self):
        """
        The str repr of DataQueue.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> repr(dq)
            '<DataQueue SCHEMA[a: ColumnType.SCALAR, b: ColumnType.QUEUE] SIZE 0>'
        """
        names = self._schema.col_names
        types = self._schema.col_types
        content = ', '.join([i + ': ' + str(j) for i, j in zip(names, types)])
        return f'<{self.__class__.__name__} SCHEMA[{content}] SIZE {self.size}>'

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
            self._size += 1
            self._not_empty.notify()
            return True

    def put_dict(self, inputs: Dict) -> bool:
        data = [inputs.get(name, None) for name in self._schema.col_names()]
        return self.put(data)

    def batch_put(self, batch_inputs: List[List]):
        assert len(batch_inputs) == self._schema.size()
        with self._not_full:
            if self._max_size > 0:
                while self.size >= self._max_size:
                    self._not_full.wait()

            inc_size = max([len(col) for col in batch_inputs])
            for col_index in range(self._schema.size()):
                if self._schema.get_col_type(col_index) == ColumnType.SCALAR:
                    self._data[col_index].put(batch_inputs[col_index][0])
                else:
                    for item in batch_inputs[col_index]:
                        self._data[col_index].put(item)
            self._size += inc_size
            self._not_empty.notify(inc_size)

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
            self._not_full.notify()
            return ret

    def get_dict(self) -> Optional[Dict]:
        data = self.get()
        if data is None:
            return None

        ret = {}
        names = self._schema.col_names()
        for i in range(len(names)):
            ret[names[i]] = data[i]
        return ret

    @property
    def size(self) -> int:
        return self._size

    @property
    def col_size(self) -> int:
        return self._schema.size()

    def clear_and_seal(self):
        with self._lock:
            self._size = 0
            self._sealed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def seal(self):
        with self._lock:
            self._sealed = True
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
            >>> dq.schema
            ['a', 'b']
        """
        return [i for i in self._schema.col_names]
    
    @property
    def type_schema(self) -> List[str]:
        """
        Return the type of queues in the DataQueue.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            >>> dq.type_schema
            [<ColumnType.SCALAR: 2>, <ColumnType.QUEUE: 1>]
        """
        return [i for i in self._schema.col_types]


class ColumnType(Enum):
    '''
    ColumnType
    '''
    QUEUE = auto()
    SCALAR = auto()


_ColumnInfo = namedtuple('_ColumnInfo', ['name', 'col_type'])


class _Schema:
    '''
    schema_info.
    '''

    def __init__(self, schema_info: List[Tuple]):
        self._cols = []
        for col in schema_info:
            self._cols.append(_ColumnInfo(*col))
        self._size = len(schema_info)

    def size(self):
        return self._size

    def col_names(self):
        return [col.name for col in self._cols]

    def col_types(self):
        return [col.col_type for col in self._cols]

    def get_col_name(self, index):
        assert index < self._size
        return self._cols[index].name

    def get_col_type(self, index):
        assert index < self._size
        return self._cols[index].col_type


class _QueueColumn:
    '''
    Queue column.
    '''

    def __init__(self):
        self._q = deque()

    def get(self):
        if len(self._q) == 0:
            return None
        return self._q.popleft()

    def put(self, data) -> bool:
        self._q.append(data)


class _ScalarColumn:
    '''
    Scalar column
    '''

    def __init__(self):
        self._data = None

    def put(self, data):
        self._data = data

    def get(self):
        return self._data
