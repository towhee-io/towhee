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

from towhee.dataframe.array.array import Array
from towhee.dataframe._schema import _Schema
from towhee.types._frame import _Frame


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
        default_cols -> provided_columns -> whats
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
        self._map_blocked = {}
        self._window_blocked = {}


        self._data_lock = threading.RLock()
        self._len = 0
        self._data_as_list = None
        self._data_as_dict = None
        self._row_iter = 0
        self._schema = None

        # TODO: Better solution for no data whatsoever
        # if no columns and no data: delay to first data added
        if columns is None and data is None:
            raise ValueError('Cannot construct dataframe without columns or initial data')

        # if column and no data: create empty arrays
        elif columns is not None and data is None:
            self._schema = _Schema()
            self._set_cols(columns)
            self._from_none()

        # if no column and data: default names and types from data
        elif columns is None and data is not None:
            self._extract_data(data)
            self._schema = _Schema()
            self._set_schema()

        # if column and data: create regular running.
        elif columns is not None and data is not None:
            self._schema = _Schema()
            self._set_cols(columns)
            self._extract_data(data)

        self._add_frame()

    def _add_frame(self):
        arr = Array(name = '_frame')
        for _ in range(self._len):
            arr.put(_Frame(self._row_iter))
            self._row_iter += 1
        self._schema.add_col('_frame', col_type=_Frame)
        self._data_as_list.append(arr)
        self._data_as_dict['_frame'] = self._data_as_list[-1]

    def _from_none(self):
        self._data_as_list = [Array(name=name) for name, _ in self._schema.cols]
        self._data_as_dict = {name: self._data_as_list[i] for i, (name, _) in enumerate(self._schema.cols)}

    def _set_cols(self, columns):
        for names, types in columns:
            self._schema.add_col(name=names, col_type = types)

    def _set_schema(self):
        if self._data_as_list[0].physical_size > 0:
            for x in self._data_as_list:
                key = x.name
                col_type = type(x.get_relative(0))
                self._schema.add_col(key, col_type)
        else:
            raise ValueError('Can\'t create df from empty data and column.')


    def _extract_data(self, data):
        # For `data` is `list`
        if isinstance(data, list):
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

    def _from_tuples(self, data):
        # check tuple length
        tuple_lengths = set(len(i) for i in data)
        if len(tuple_lengths) == 1:
            tuple_length = tuple_lengths.pop()
        else:
            raise ValueError('can not construct DataFrame from unequal-length tuples')

        # create arrays
        if self._schema is None:
            self._data_as_list = [Array(name='Col_' + str(i)) for i in range(tuple_length)]
            self._data_as_dict = {'Col_' + str(i): self._data_as_list[i] for i in range(tuple_length)}
        else:
            # check columns length
            if self._schema.col_count != tuple_length:
                raise ValueError('length of columns is not equal to the length of tuple')
            self._data_as_list = [Array(name=name) for name, _ in self._schema.cols]
            self._data_as_dict = {name: self._data_as_list[i] for i, (name, _) in enumerate(self._schema.cols)}

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

        self._data_as_list = []
        self._data_as_dict = {}

        if self._schema is None:
            for i, arr in enumerate(data):
                if arr.name is None:
                    arr.set_name('Col_' + str(i))
                    self._data_as_list.append(arr)
                    self._data_as_dict['Col_' + str(i)] = self._data_as_list[-1]
                else:
                    self._data_as_list.append(arr)
                    self._data_as_dict[arr.name] = self._data_as_list[-1]

        else:
            # check columns length
            if self._schema.col_count != len(data):
                raise ValueError('length of columns is not equal to the number of arrays')
            for i, arr in enumerate(data):
                arr.set_name(self._schema.col_key(i))
                self._data_as_list.append(arr)
                self._data_as_dict[self._schema.col_key(i)] = self._data_as_list[-1]

    def _from_dict(self, data):
        # check dict values
        vals = set(type(array) for array in data.values())
        if len(vals) != 1 or not vals.pop() == Array:
            raise ValueError('value type in data should be towhee.dataframe.Array')

        # check arrays length
        array_lengths = set(len(array) for array in data.values())
        if len(array_lengths) != 1:
            raise ValueError('arrays in data should have equal length')

        self._len = array_lengths.pop()

        self._data_as_list = []
        self._data_as_dict = {}

        if self._schema is None:
            for key, val in data.items():
                val.set_name(key)
                val.append()
                self._data_as_list.append(val)
                self._data_as_dict[key] = self._data_as_list[-1]
            
        else:
            for i, arr in enumerate(data.values()):
                arr.set_name(self._schema.col_key(i))
                self._data_as_list.append(arr)
                self._data_as_dict[self._schema.col_key(i)] = self._data_as_list[-1]
        

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
            print(self.physical_size)
            for x in range(self._min_offset, self._min_offset + self.physical_size):
                values = []
                print(x)
                for i in range(len(self._data_as_list)):
                    val = self._data_as_list[i][x]
                    if type(val) == _Frame:
                        val = (val.row_id, val.timestamp)
                    values.append(str(val))
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
            return [x for x, _ in self._schema.cols]

    @property
    def types(self) -> List[Any]:
        return [y for _, y in self._schema.cols]


    @property
    def sealed(self) -> bool:
        with self._data_lock:
            return self._sealed

    def get_window(self, offset, cutoff, iter_id):

        with self._iterator_lock:
            if iter_id is not None and self._iterators[iter_id] == float('inf'):
                return Responses.KILLED, None

        if self._schema is None:
            raise ValueError('Schema not set.')

        col = self._schema.col_index('_frame')

        if col is None:
            raise ValueError('Not a column.')

        if self._schema.col_type(col) != _Frame:
            raise ValueError('Column not a frame column.')

        with self._data_lock:
            if offset < self._min_offset:
                return Responses.INDEX_GC, None

            ret = []
            count = 0
            # cutoff = ('timestamp' or 'row_id', int)
            for x in range(offset, self._len):
                if cutoff[0] == 'row_id':
                    if self.__getitem__(x)[col].row_id < cutoff[1]:
                        ret.append(self.__getitem__(x))
                        count += 1
                    else:
                        break
                # elif cutoff[0] == 'timestamp':
                #     if self.__getitem__(x)[col].timestamp < cutoff[1]:
                #         ret.append(self.__getitem__(x))
                #         count += 1
                #     if self.__getitem__(x)[col].timestamp == cutoff[1]:
                #         filled=True

            # Window valid but not fufilled by last value.
            if offset + count == self._len and not self._sealed:
                # Line if window doesnt need to wait for all if not blocking
                # return Responses.WINDOW_NOT_DONE, ret
                return Responses.WINDOW_NOT_DONE, None
            elif offset + count == self._len and self._sealed:
                return Responses.APPROVED_DONE, ret
            # Window fufilled.
            elif offset + count <= self._len:
                return Responses.APPROVED_CONTINUE, ret
            else:
                return Responses.UNKOWN_ERROR, None


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

            self._data_as_dict['_frame'].put(_Frame(self._row_iter))
            self._row_iter += 1
            self._len += 1
            cur_len = self._len
            frame = self._data_as_dict['_frame'][-1]

        with self._iterator_lock:
            if len(self._map_blocked) > 0:
                id_event = self._map_blocked.pop(cur_len, None)
                if id_event is not None:
                    for _, event in id_event:
                        event.set()
            if len(self._window_blocked) > 0:
                rem = []
                for cutoff, id_events in self._window_blocked.items():
                    if cutoff[1] <= getattr(frame, cutoff[0]):
                        for _, event in id_events:
                            event.set()
                        rem.append(cutoff)
                for key in rem:
                    del self._window_blocked[key]

    def _put_list(self, item: list):
        assert len(item) == self._schema.col_count - 1

        # I believe its faster to loop through and check than list comp
        for i, x in enumerate(item):
            assert isinstance(x, self._schema.col_type(i))

        for i, x in enumerate(item):
            self._data_as_list[i].put(x)

    def _put_tuple(self, item: tuple):
        assert len(item) == self._schema.col_count - 1

        # I believe its faster to loop through and check than list comp
        for i, x in enumerate(item):
            assert isinstance(x, self._schema.col_type(i))

        for i, x in enumerate(item):
            self._data_as_list[i].put(x)

    def _put_dict(self, item: dict):
        assert len(item) == self._schema.col_count - 1

        # I believe its faster to loop through and check than list comp
        for key, val in item.items():
            assert isinstance(val, self._schema.col_type(self._schema.col_index(key)))

        for key, val in item.items():
            self._data_as_list[self._schema.col_index(key)].put(val)

    def seal(self):
        with self._data_lock:
            self._sealed = True
            for _, id_event in self._map_blocked.items():
                for _, event in id_event:
                    event.set()
            self._map_blocked.clear()
            for _, id_event in self._window_blocked.items():
                for _, event in id_event:
                    event.set()
            self._window_blocked.clear()

    def gc(self):
        with self._data_lock:
            self._min_offset = min([value for _, value in self._iterators.items()])
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
        with self._iterator_lock:
            if self._iterators[iter_id] <= (offset + 1):
                self._iterators[iter_id] = offset + 1
        self.gc()

    def notify_map_block(self, iter_id, event, offset, count):
        with self._iterator_lock:
            index = offset + count
            if self._map_blocked.get(index) is None:
                self._map_blocked[index] = []
            self._map_blocked[index].append((iter_id, event))
            return True
    
    def notify_window_block(self, iter_id, event, cutoff):
        # cutoff = ('timestamp' or 'row_id', int)
        with self._iterator_lock:
            if self._window_blocked.get(cutoff) is None:
                self._window_blocked[cutoff] = []
            self._window_blocked[cutoff].append((iter_id, event))



    # TODO kill blocked iters or kill all iters?
    def unblock_iters(self):
        with self._iterator_lock:
            for _, id_event in self._map_blocked.items():
                for it_id, event in id_event:
                    self._iterators[it_id] = float('inf')
                    event.set()
            for _, id_event in self._window_blocked.items():
                for it_id, event in id_event:
                    self._iterators[it_id] = float('inf')
                    event.set()


class Responses(Enum):
    INDEX_GC = 1
    INDEX_OOB_UNSEALED = 2
    INDEX_OOB_SEALED = 3
    APPROVED_CONTINUE = 4
    APPROVED_DONE = 5
    KILLED = 6
    UNKOWN_ERROR = 7
    WINDOW_NOT_DONE = 8
    
