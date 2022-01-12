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
from enum import Enum
from typing import List, Tuple, Any

from towhee.dataframe.array import Array


class DataFrame:
    """
    A `DataFrame` is a collection of immutable, potentially heterogeneous blogs of data.

    Args:
        columns (`list[Tuple[str, Any]]`)
            The list of the column names and their corresponding data types.
        name (`str`):
            Name of the dataframe; `DataFrame` names should be the same as its
            representation.
        data (`list[towhee.dataframe.Array]` or `list[Tuple]` or `dict[str, towhee.dataframe.Array]`):
            The data of the `DataFrame`. Internally, the data will be organized
            in a column-based manner.
    """

    def __init__(
        self,
        columns: List[Tuple[str, Any]] = None,
        name: str = None,
        data=None,
    ):
        self._name = name
        self._sealed = False

        self._iterator_lock = threading.RLock()
        self._it_id = 0
        self._iterators = {}
        self._min_offset = 0

        self._block_lock = threading.RLock()
        self._blocked = {}

        self._data_lock = threading.RLock()
        self._len = 0
        self._data_as_list = None
        self._data_as_dict = None

        # TODO: Enforce columns everytime except for when dict passed in.
        # or supply default 'col_x' naming convention.
        # TODO: This is equivalent to @junjie's schema
        if columns is not None:
            self._types = {x[0]: x[1] for x in columns}
            self._columns = [x[0] for x in columns]

        elif not isinstance(data, dict):
            raise ValueError(
                    'Cannot construct dataframe without colum names and types (except for dict).')

        # For `data` is empty
        if not data:
            self._from_none()

        # For `data` is `list`
        elif isinstance(data, list):
            container_types = set(type(i) for i in data)
            if len(container_types) != 1:
                raise ValueError(
                    'can not construct Dataframe from a list of hybrid data containers. Try list[Tuple] or list[Array].')
            container_type = container_types.pop()

            # For `data` is `list[tuple]`
            if container_type is tuple:
                self._from_tuples(data)
            # For `data` is `list[towhee.dataframe.Array]`
            elif container_type is Array:
                self._from_arrays(data)
            else:
                raise ValueError('can not construct DataFrame from list[%s]' % (container_type))

        # For `data` is `dict`
        elif isinstance(data, dict):
            self._from_dict(data)

        # Unrecognized data types
        else:
            raise ValueError('can not construct DataFrame from data type %s' % (type(data)))

    def __getitem__(self, key):
        with self._data_lock:
            # access a row
            if isinstance(key, int):
                return tuple(self._data_as_list[i][key] for i in range(len(self._data_as_list)))
            # access a column
            elif isinstance(key, str):
                return self._data_as_dict[key]

    def __str__(self):
        """
        Simple to_string for printing and debugging dfs. Currently assumes that the data can be str()'ed.
        """
        ret = ''
        formater = ''
        columns = []
        with self._data_lock:
            for x in range(len(self._data_as_list)):
                columns.append(self._data_as_list[x].name)
                formater += '{' + str(x) + ':30}'
            ret += formater.format(*columns) + '\n'

            for x in range(self._min_offset, self._min_offset + self.physical_size):
                values = []
                for i in range(len(self._data_as_list)):
                    values.append(str(self._data_as_list[i][x]))
                ret += formater.format(*values) + '\n'

            return ret

    def __len__(self):
        with self._data_lock:
            return self._len

    @property
    def physical_size(self):
        with self._data_lock:
            if self._data_as_list is None:
                return 0
            return self._data_as_list[0].physical_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def iterators(self) -> List[int]:
        with self._iterator_lock:
            return self._iterators

    @property
    def data(self) -> List[Array]:
        with self._data_lock:
            return self._data_as_list

    @property
    def columns(self) -> List[str]:
        with self._data_lock:
            return self._columns

    @property
    def types(self) -> List[Any]:
        return self._types


    @property
    def sealed(self) -> bool:
        with self._data_lock:
            return self._sealed

    def window_get(self, offset, start, end, col, iter_id):
        raise NotImplementedError

    def get(self, offset, count = 1, iter_id = None):
        """
        Dataframe's function to return data at current offset and count.

        Args:
            offset:
                The index to get data from.
            count:
                How many rows to return
            iter_id:

        """
        with self._iterator_lock:
            if iter_id is not None and self._iterators[iter_id] == float('inf'):
                return Responses.KILLED, None

        with self._data_lock:
            if offset < self._min_offset:
                return Responses.INDEX_GC, None

            elif offset + count <= self._len:
                return Responses.APPROVED_CONTINUE, [self.__getitem__(x) for x in range(offset, offset + count)]

            elif self._sealed:
                if self._len <= offset:
                    return Responses.INDEX_OOB_SEALED, None
                else:
                    return Responses.APPROVED_DONE, [self.__getitem__(x) for x in range(offset, self._len)]

            elif offset + count >= self._len:
                return Responses.INDEX_OOB_UNSEALED, None # [self.__getitem__(x) for x in range(offset, self._len)]

            else:
                return Responses.UNKOWN_ERROR, None

    def put(self, item) -> None:
        """Put values into dictionary

        For now it takes:
        tuple
        towhee array
        dict(requires col names)
        """
        assert not self._sealed, f'DataFrame {self._name} is already sealed, can not put data'
        assert isinstance(item, (tuple, dict, list)), 'Dataframe needs to be of type (tuple, dict, list), not %s' % (type(item))
        with  self._data_lock:
            if isinstance(item, list):
                self._put_list(item)
            elif isinstance(item, dict):
                self._put_dict(item)
            else: # type(item) is tuple:
                self._put_tuple(item)

            self._len += 1
            cur_len = self._len

        with self._iterator_lock:
            cv, _ = self._blocked.pop(cur_len, (None, None))
            if cv is not None:
                with cv:
                    cv.notify_all()

    def _put_list(self, item: list):
        assert len(item) == len(self._types)

        # I believe its faster to loop through and check than list comp
        for i, x in enumerate(item):
            assert isinstance(x, self._types[self._columns[i]])

        for i, x in enumerate(item):
            self._data_as_list[i].put(x)

    def _put_tuple(self, item: tuple):
        assert len(item) == len(self._types)

        # I believe its faster to loop through and check than list comp
        for i, x in enumerate(item):
            assert isinstance(x, self._types[self._columns[i]])

        for i, x in enumerate(item):
            self._data_as_list[i].put(x)

    def _put_dict(self, item: dict):
        assert len(item) == len(self._types)

        # I believe its faster to loop through and check than list comp
        for key, val in item.items():
            assert isinstance(val, self._types[key])

        for key, val in item.items():
            self._data_as_list[self._columns.index(key)].put(val)

    def _from_none(self):
        self._data_as_list = [Array(name=self._columns[i]) for i in range(len(self._columns))]
        self._data_as_dict = {self._columns[i]: self._data_as_list[i] for i in range(len(self._columns))}

    def _from_tuples(self, data):
        # check tuple length
        tuple_lengths = set(len(i) for i in data)
        if len(tuple_lengths) == 1:
            tuple_length = tuple_lengths.pop()
        else:
            raise ValueError('can not construct DataFrame from unequal-length tuples')

        # check columns length
        if self._columns and len(self._columns) != tuple_length:
            raise ValueError('length of columns is not equal to the length of tuple')

        # create arrays
        if self._columns:
            self._data_as_list = [Array(name=self._columns[i]) for i in range(tuple_length)]
            self._data_as_dict = {self._columns[i]: self._data_as_list[i] for i in range(tuple_length)}
        else:
            self._data_as_list = [Array()] * tuple_length
            self._data_as_dict = None

        # tuples to arrays
        for row in data:
            for i, element in enumerate(row):
                self._data_as_list[i].put(element)

        self._len = len(data)

    def _from_arrays(self, data):
        # check array length
        array_lengths = set(len(array) for array in data)
        if len(array_lengths) != 1:
            raise ValueError('arrays in data should have equal length')

        self._len = array_lengths.pop()

        # check columns length
        if self._columns and len(self._columns) != len(data):
            raise ValueError('length of columns is not equal to the number of arrays')

        self._data_as_list = data

        if self._columns:
            self._data_as_dict = {self._columns[i]: self._data_as_list[i] for i in range(len(data))}
        else:
            self._data_as_dict = None

    def _from_dict(self, data):
        # check dict values
        self._types = {}
        self._columns = []
        for key, value in data.items():
            self._types[key[0]] = key[1]
            self._columns.append(key[0])
            if not isinstance(value, Array):
                raise ValueError('value type in data should be towhee.dataframe.Array')

        # check arrays length
        array_lengths = set(len(array) for array in data.values())
        if len(array_lengths) != 1:
            raise ValueError('arrays in data should have equal length')

        self._len = array_lengths.pop()
        # TODO: Check if data lines up by just converting to list
        self._data_as_list = list(data.values())
        self._data_as_dict = {key[0]: val for key, val in data.items()}

    def seal(self):
        with self._data_lock:
            self._sealed = True
            for _, (cv, _) in self._blocked.items():
                with cv:
                    cv.notify_all()
            self._blocked.clear()

    def gc(self):
        with self._data_lock:
            self._min_offset = min([value for _, value in self._iterators.items()])
            if self._min_offset == float('inf'):
                raise ValueError('All iterators killed')

            for x in self._data_as_list:
                x.gc(self._min_offset)

    def register_iter(self):
        with self._iterator_lock:
            self._it_id += 1
            self._iterators[self._it_id] = 0
            return self._it_id

    def ack(self, iter_id, offset):
        """
        To notice the DataFrame that the iterated rows has been successfully processed.

        An acknowledgement (ack) will notice the `DataFrame`s that the rows already
        iterated over are no longer used, and can be deleted from the system.

        Args:
            iter_id (`int`):
                The iterator id to set offset for.
            offset (`int`):
                The latest accepted offset.
        """
        # No lock needed since iterators only deal with their index
        with self._iterator_lock:
            if self._iterators[iter_id] <= (offset + 1):
                self._iterators[iter_id] = offset + 1
        self.gc()

    def notify_block(self, iter_id, offset, count):
        with self._iterator_lock:
            index = offset + count
            if self._blocked.get(index) is None:
                self._blocked[index] = (threading.Condition(), [])
            self._blocked[index][1].append(iter_id)
            return self._blocked[index][0]
    
    # TODO kill blocked iters or kill all iters?
    def unblock_iters(self):
        with self._iterator_lock:
            for _, (cv, iters) in self._blocked.items():
                for ite in iters:
                    self._iterators[ite] = float('inf')
                with cv:
                    cv.notify_all()


class Responses(Enum):
    INDEX_GC = 1
    INDEX_OOB_UNSEALED = 2
    INDEX_OOB_SEALED = 3
    APPROVED_CONTINUE = 4
    APPROVED_DONE = 5
    KILLED = 6
    UNKOWN_ERROR = 7
