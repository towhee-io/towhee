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
        # self._data_as_dict = None
        self._schema = None

        # TODO: Better solution for no data whatsoever
        # if no columns and no data: delay to first data added

        if columns is None and data is None:
            raise ValueError('Cannot construct dataframe without columns or initial data')
        elif isinstance(data, dict):
            self._extract_data(data)
            self._schema = _Schema()
            self._set_schema()
            self._add_frame(init = True)

        # if column and no data: create empty arrays
        elif columns is not None and data is None:
            self._schema = _Schema()
            self._set_schema(columns)
            self._from_none()

        # if no column and data: default names and types from data
        elif columns is None and data is not None:
            self._extract_data(data)
            self._schema = _Schema()
            self._set_schema()
            self._add_frame(init = True)

        # if column and data: create regular running.
        elif columns is not None and data is not None:
            self._schema = _Schema()
            self._set_schema(columns)
            self._extract_data(data)

    def _add_frame(self, init = False):
        """Adds/corrects frame data."""

        col_index = self._schema.col_index('_frame')
        created = False

        if col_index == len(self._data_as_list):
            arr = Array(name = '_frame')
            self._data_as_list.append(arr)
            created = True

        if init:
            for x in range(self._len):
                if not created:
                    self._data_as_list[col_index][x].row_id = x
                else:
                    self._data_as_list[col_index].put(_Frame(x))

    def _from_none(self):
        """Initialize empty Arrays."""
        self._data_as_list = [Array(name=name) for name, _ in self._schema.cols]

    def _set_schema(self, columns = None):
        """Set the columns in the schema."""
        if columns is not None:
            for names, types in columns:
                self._schema.add_col(name=names, col_type = types)
            if self._schema.col_index('_frame') is None:
                    self._schema.add_col('_frame', _Frame)
        else:
            if self._data_as_list[0].physical_size > 0:
                for x in self._data_as_list:
                    key = x.name
                    col_type = type(x.get_relative(0))
                    self._schema.add_col(key, col_type)
                if self._schema.col_index('_frame') is None:
                    self._schema.add_col('_frame', _Frame)
            else:
                raise ValueError('Can\'t create df from empty data and no columns.')

    def _extract_data(self, data):
        """Convert the passed in data and store it."""
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
        """Convert passed in data from tuples to Arrays"""
        # check tuple length
        tuple_lengths = set(len(i) for i in data)
        if len(tuple_lengths) == 1:
            tuple_length = tuple_lengths.pop()
        else:
            raise ValueError('can not construct DataFrame from unequal-length tuples')

        # create arrays
        if self._schema is None:
            self._data_as_list = [Array(name='Col_' + str(i)) for i in range(tuple_length)]
            for row in data:
                for i, element in enumerate(row):
                    self._data_as_list[i].put(element)
            self._len = len(data)
        else:
            self._data_as_list = [Array(name=name) for name, _ in self._schema.cols]
            for row in data:
                self.put(row)

    def _from_arrays(self, data):
        """Convert passed in data from arrays to Arrays"""
        # check array length
        array_lengths = set(len(array) for array in data)
        if len(array_lengths) != 1:
            raise ValueError('arrays in data should have equal length')

        self._data_as_list = []

        if self._schema is None:
            for i, arr in enumerate(data):
                if arr.name is None:
                    arr.set_name('Col_' + str(i))
                    self._data_as_list.append(arr)
                else:
                    self._data_as_list.append(arr)
            self._len = array_lengths.pop()

        else:
            self._data_as_list = [Array(name=name) for name, _ in self._schema.cols]
            for x in range(array_lengths.pop()):
                current_line = [i[x] for i in data]
                self.put(current_line)

    def _from_dict(self, data):
        """Convert passed in data from dict to Arrays"""
        vals = set(type(array) for array in data.values())
        if len(vals) != 1 or not vals.pop() == Array:
            raise ValueError('value type in data should be towhee.dataframe.Array')

        # check arrays length
        array_lengths = set(len(array) for array in data.values())
        if len(array_lengths) != 1:
            raise ValueError('arrays in data should have equal length')

        self._data_as_list = []

        for key, val in data.items():
            val.set_name(key)
            self._data_as_list.append(val)
        self._len = array_lengths.pop()

    def __getitem__(self, key):
        """Get data at the passed in offset."""
        with self._data_lock:
            # access a row
            if isinstance(key, int):
                return tuple(self._data_as_list[i][key] for i in range(len(self._data_as_list)))
            # access a column
            elif isinstance(key, str):
                index = self._schema.col_index(key)
                return self._data_as_list[index]

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
                    val = self._data_as_list[i][x]
                    if isinstance(val, _Frame):
                        val = (val.row_id, val.timestamp)
                    values.append(str(val))
                ret += formater.format(*values) + '\n'

            return ret

    def __len__(self):
        with self._data_lock:
            return self._len

    @property
    def physical_size(self):
        """The number of elements left in the Dataframe."""
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
        """
        Window based data retrieval. Windows include everything up to but not including
        the cutoff.

        Args:
            offset (`int`):
                The starting index to read data from.
            cutoff ("timestamp" | "row_id", `int`):
                The current upper window limit.
            iter_id (`int`):
                Which iterator is reading the data.
        Raises:
            ValueError:
                Incorrect _Frame column.
        Returns:
            Responses:
                Different states of response codes depending on the dataframe state.
        """

        # When the iterator is forcefully unblocked using unblock_iters().
        with self._iterator_lock:
            if iter_id is not None and self._iterators[iter_id] == float('inf'):
                return Responses.KILLED, None

        # If there is no schema that means that initialization broke at some point.
        if self._schema is None:
            raise ValueError('Schema not set.')

        # TODO: Maybe remove if _frame column is always the last one.
        col = self._schema.col_index('_frame')

        # If there is no _frame column something has broken.
        if col is None:
            raise ValueError('Not a column.')

        # If the '_frame' column doesnt hold _Frames then something is broken.
        if self._schema.col_type(col) != _Frame:
            raise ValueError('Column not a frame column.')

        with self._data_lock:
            # Up to iterators to decide what to do if the value they want is GCed.
            if offset < self._min_offset:
                return Responses.INDEX_GC, None

            ret = []
            count = 0
            for x in range(offset, self._len):
                if cutoff[0] == 'row_id':
                    if self.__getitem__(x)[col].row_id < cutoff[1]:
                        ret.append(self.__getitem__(x))
                        count += 1
                    else:
                        break
                elif cutoff[0] == 'timestamp':
                    if self.__getitem__(x)[col].timestamp < cutoff[1]:
                        ret.append(self.__getitem__(x))
                        count += 1
                    else:
                        break

            # Window valid but not fufilled by last value.
            if offset + count == self._len and not self._sealed:
                return Responses.WINDOW_NOT_DONE, None

            # Window valid, finished and sealed.
            elif offset + count == self._len and self._sealed:
                return Responses.APPROVED_DONE, ret

            # Window fufilled and available next window.
            elif offset + count <= self._len:
                return Responses.APPROVED_CONTINUE, ret

            # Something broke.
            else:
                return Responses.UNKOWN_ERROR, None

    def get(self, offset, count = 1, iter_id = None):
        """
        Get data from dataframe based on offset and how many values requested.

        Args:
            offset (`int`):
                The starting index to read data from.
            count (`int`):
                How many values to retrieve.
            iter_id (`int`):
                Which iterator is reading the data.

        Returns:
            Responses:
                Different states of response codes depending on the dataframe state.
        """

        # When the iterator is forcefully unblocked using unblock_iters().
        with self._iterator_lock:
            if iter_id is not None and self._iterators[iter_id] == float('inf'):
                return Responses.KILLED, None

        with self._data_lock:
            # Up to iterators to decide what to do if the value they want is GCed.
            if offset < self._min_offset:
                return Responses.INDEX_GC, None

            # The batch of data is available and there is more available.
            elif offset + count <= self._len:
                return Responses.APPROVED_CONTINUE, [self.__getitem__(x) for x in range(offset, offset + count)]

            elif self._sealed:
                # If no more data is left and trying to get more, iterator will stop and no data returned.
                if self._len <= offset:
                    return Responses.INDEX_OOB_SEALED, None

                # If dataframe is sealed, remaining values will be returned.
                else:
                    return Responses.APPROVED_DONE, [self.__getitem__(x) for x in range(offset, self._len)]

            # If the requested offset is out of bounds but dataframe is still writing data.
            elif offset + count >= self._len:
                return Responses.INDEX_OOB_UNSEALED, None

            # Something broke.
            else:
                return Responses.UNKOWN_ERROR, None

    def put(self, item) -> None:
        """
        Append a new value to the dataframe.

        Args:
            item (dict | list | Array | tuple):
                The row to insert.

        """
        assert not self._sealed, f'DataFrame {self._name} is already sealed, can not put data'
        assert isinstance(item, (tuple, dict, list, Array)), 'Dataframe needs to be of type (tuple, dict, list), not %s' % (type(item))
        with  self._data_lock:
            if isinstance(item, list) or isinstance(item, Array):
                self._put_list(item)
            elif isinstance(item, dict):
                self._put_dict(item)
            else:
                self._put_tuple(item)

            frame = self._data_as_list[self._schema.col_index('_frame')][-1]
            self._len += 1
            cur_len = self._len

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
        """Appending a list to the dataframe."""

        # If frame is there, simply change its row_id.
        if len(item) == self._schema.col_count:
            item[self._schema.col_index('_frame')].row_id = self._len

        # If frame is not there, assume that the dataframe added it to end of schema on init.
        # If that is not the case then the inputted data is wrong and the assert will be
        # triggered.
        else:
            item.append((_Frame(self._len)))

        for i, x in enumerate(item):
            assert isinstance(x, self._schema.col_type(i))
        
        for i, x in enumerate(item):
            self._data_as_list[i].put(x)

    def _put_tuple(self, item: tuple):
        """Appending a tuple to the dataframe."""
        
        item = list(item)

        if len(item) == self._schema.col_count:
            item[self._schema.col_index('_frame')].row_id = self._len

        else:
            item.append((_Frame(self._len)))

        for i, x in enumerate(item):
            assert isinstance(x, self._schema.col_type(i))

        for i, x in enumerate(item):
            self._data_as_list[i].put(x)

    def _put_dict(self, item: dict):
        """Appending a dict to the dataframe."""

        if len(item) == self._schema.col_count:
            item['_frame'].row_id = self._len
        else:
            item['_frame'] = _Frame(self._len)

        # I believe its faster to loop through and check than list comp
        for key, val in item.items():
            assert isinstance(val, self._schema.col_type(self._schema.col_index(key)))

        for key, val in item.items():
            self._data_as_list[self._schema.col_index(key)].put(val)

    def seal(self):
        """
        Function to seal the dataframe. A sealed dataframe will not accept anymore data and
        the act of sealing the dataframe will trigger blocked iterators to pull the remaining data.
        """
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
        """Internal garbage collection function to trigger towhee.Array GC."""
        with self._data_lock:
            self._min_offset = min([value for _, value in self._iterators.items()])
            for x in self._data_as_list:
                x.gc(self._min_offset)

    def register_iter(self):
        """
        Registering an iter allows the df to keep track of where each iterator is in order
        to coordinate garbage collection and other processes.

        Returns:
            An iterator ID that is used for a majority of iterator-dataframe interaction.
        """
        with self._iterator_lock:
            self._it_id += 1
            self._iterators[self._it_id] = 0
            return self._it_id

    def ack(self, iter_id, offset):
        """
        An acknowledgement (ack) will notice the `DataFrame`s that the rows already
        iterated over are no longer used, and can be garbage collected.

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
        """
        Used by map based iterators to notify the dataframe that it is waiting on a value.

        Args:
            iter_id (`int`):
                The iterator id of the blocked iterator.
            event (`threading.Event`):
                The event to trigger once the value being waited on is available.
            offset (`int`):
                The offset that is being waited on.
            count (`int):
                How many values are being waited on.
        """
        with self._iterator_lock:
            index = offset + count
            if self._map_blocked.get(index) is None:
                self._map_blocked[index] = []
            self._map_blocked[index].append((iter_id, event))
            return True

    def notify_window_block(self, iter_id, event, cutoff):
        """
        Used by window based iterators to notify the dataframe that it is waiting on a value.

        Args:
            iter_id (`int`):
                The iterator id of the blocked iterator.
            event (`threading.Event`):
                The event to trigger once the value being waited on is available.
            cutoff ("timestamp" | "row_id", `int`):
                The window cutoff that is is being waited on.
        """
        with self._iterator_lock:
            if self._window_blocked.get(cutoff) is None:
                self._window_blocked[cutoff] = []
            self._window_blocked[cutoff].append((iter_id, event))



    # TODO kill blocked iters or kill all iters?
    def unblock_iters(self):
        """
        Forceful unblocking/killing of the blocked iters.
        """
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
