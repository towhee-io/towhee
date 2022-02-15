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


from towhee.dataframe.variable import Variable
from towhee.dataframe.array.array import Array
from towhee.dataframe._schema import _Schema
from towhee.types._frame import _Frame, FRAME


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
        name: str,
        columns: List[Tuple[str, str]],
    ):
        self._name = name
        self._sealed = False

        self._iterator_lock = threading.RLock()
        self._it_id = 0
        self._iterators = {}
        self._min_offset = 0

        self._block_lock = threading.RLock()
        self._map_blocked = {}
        self._window_start_blocked = {}
        self._window_end_blocked = {}

        self._data_lock = threading.RLock()
        self._len = 0
        self._data_as_list = []
        self._schema = _Schema()

        self._initialize_storage(columns)

    def _initialize_storage(self, columns):
        """Set the columns in the schema."""
        for name, col_type in columns:
            self._schema.add_col(name=name, col_type = col_type)
        self._schema.add_col(FRAME, '_Frame')
        self._data_as_list = [Array(name=name) for name, _ in self._schema.cols]

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

            if self._min_offset != float('inf'):
                for x in range(self._min_offset, self._min_offset + self.physical_size):
                    values = []
                    for i in range(len(self._data_as_list)):
                        val = self._data_as_list[i][x]
                        if isinstance(val, _Frame):
                            val = (val.row_id, val.prev_id, val.timestamp,)
                        values.append(str(val))
                    ret += formater.format(*values) + '\n'

            return ret

    def __len__(self):
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
        return self._iterators

    @property
    def data(self) -> List[Array]:
        with self._data_lock:
            return self._data_as_list

    @property
    def columns(self) -> List[str]:
        return [x for x, _ in self._schema.cols]

    @property
    def types(self) -> List[Any]:
        return [y for _, y in self._schema.cols]

    @property
    def sealed(self) -> bool:
        return self._sealed

    def get_window(self, start, end, step, comparator, iter_id):
        """
        Window based data retrieval. Windows include everything up to but not including
        the cutoff.

        Args:
            start (`int`):
                The starting index to read data from.
            end (`int`):
                The current upper window limit.
            step (`int`):
                How far the window will slide for next call
            comparator ('timestamp' | 'row_id'):
                Which value is being used for window calculations.
            iter_id (`int`):
                Which iterator is reading the data.

        Returns:
            Responses:
                Different states of response codes depending on the dataframe state.

        Situations:

        Unsealed:
            [0, 1, 2, 3] window = (5, 10)
            1. Window start is greater than the latest value in df:
                a. block for value greater than window start

            [0, 1, 2, 3] window = (0, 10)
            2. Window end is greater than the latest value in df:
                a. block for value that is greater than window end

            [5, 6, 7, 8] window = (1, 4)
            3. Window end is smaller than all values in df
                a. return None and continue to next window

            [0, 1, 3, 4] window = (0, 3)
            4. Values exist between window start and window end:
                a. return values, clear data up to first value

        Sealed:
            [0, 1, 2, 3] window = (5, 10)
            1. Window start is greater than the latest value in df:
                a. Stop iteration

            [0, 1, 2, 3] window = (0, 10)
            2. Window end is greater than the latest value in df:
                a. return remaining window values and stop iteration

            [5, 6, 7, 8] window = (1, 4)
            3. Window end is smaller than all values in df
                a. return None and continue to next window

            [0, 1, 3, 4] window = (0, 3)
            4. Values exist between window start and window end:
                a. return values, clear data up to first value

        """

        # When the iterator is forcefully unblocked using unblock_iters().
        with self._iterator_lock:
            if iter_id is not None and self._iterators.get(iter_id) is None:
                return Responses.KILLED, None, None

        base_offset = self._iterators[iter_id]
        rets = []
        min_offset = None
        max_offset = base_offset

        with self._data_lock:
            #  If the df is empty, we dont want to do anything but wait for a next value if unsealed.
            if self.physical_size == 0:
                if self.sealed:
                    return Responses.EMPTY_SEALED, None, None
                else:
                    return Responses.EMPTY, None, None
            #  If the first element in the df is larger than the window end we want to move to the next viable window
            if getattr(self._data_as_list[-1][base_offset], comparator) >= end:
                goal = getattr(self._data_as_list[-1][base_offset], comparator)
                new_start = start + step * ((goal - start)//step)
                return Responses.OLD_WINDOW, None, (new_start, (new_start - start) + end)
            #  If the last element of the df is smaller than the start of the window, proceed to next windows.
            elif getattr(self._data_as_list[-1][-1], comparator) < start:
                if self.sealed:
                    return Responses.FUTURE_WINDOW_SEALED, None, None
                else:
                    return Responses.FUTURE_WINDOW, None, self._len

            # iterating through values to see which ones fall in the window. min offset = first value, max offset = last value
            for i in range(base_offset, self._len):
                if getattr(self._data_as_list[-1][i], comparator) >= start and getattr(self._data_as_list[-1][i], comparator) < end:
                    if min_offset is None:
                        min_offset = i
                    max_offset = i
                    rets.append(self.__getitem__(max_offset))
                if getattr(self._data_as_list[-1][i], comparator) >= end:
                    break

        # If the max offset is the length of the df
        if max_offset == self._len - 1 and self.sealed:
            # Need to check if there is a next window that fits, if so, continue
            if min_offset + step < self._len:
                return Responses.APPROVED_CONTINUE, rets, min_offset + step
            # If not, return whats left and have iterator close.
            else:
                return Responses.APPROVED_DONE, rets, max_offset
        # If the last value that fits in window is also the last value of df, cant close
        # window yet as we dont know if the next value fits in the same window.
        elif max_offset == self._len - 1 and not self.sealed:
            return Responses.WINDOW_NOT_DONE, None, None
        else:
            return Responses.APPROVED_CONTINUE, rets, min_offset + step


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
            if iter_id is not None and self._iterators.get(iter_id) is None:
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
        assert isinstance(item, (tuple, dict, list)), f'Dataframe input must be of type (tuple, dict, list), not {type(item)}'
        with  self._data_lock:
            if not isinstance(item, list):
                item = [item]
            for x in item:
                if isinstance(x, dict):
                    self._put_dict(x)
                elif isinstance(x, tuple):
                    self._put_tuple(x)
                else:
                    raise ValueError('Input data is of wrong format.')

            frame = self._data_as_list[-1][-1].value
            cur_len = self._len

        # Release blocked iterators if their criteria met.
        with self._block_lock:
            if len(self._map_blocked) > 0:
                for i, id_events in self._map_blocked.items():
                    if i <= cur_len:
                        for _, event in id_events:
                            event.set()

            if len(self._window_start_blocked) > 0:
                rem = []
                for cutoff, id_events in self._window_start_blocked.items():
                    if cutoff[1] <= getattr(frame, cutoff[0]):
                        for _, event in id_events:
                            event.set()
                        rem.append(cutoff)
                for key in rem:
                    del self._window_start_blocked[key]

            if len(self._window_end_blocked) > 0:
                rem = []
                for cutoff, id_events in self._window_end_blocked.items():
                    if cutoff[1] < getattr(frame, cutoff[0]):
                        for _, event in id_events:
                            event.set()
                        rem.append(cutoff)
                for key in rem:
                    del self._window_end_blocked[key]


    def _put_tuple(self, item: tuple):
        """Appending a tuple to the dataframe."""

        item = list(item)

        if isinstance(item[-1], Variable) and not isinstance(item[-1].value, _Frame):
            item.append(Variable(FRAME, _Frame(row_id=self._len)))
        else:
            item[-1].value.row_id = self._len

        for i, x in enumerate(item):
            assert isinstance(x, Variable)
            assert str(x.value.__class__.__name__) == self._schema.col_type(i)

        for i, x in enumerate(item):
            self._data_as_list[i].put(x)

        self._len += 1

    def _put_dict(self, item: dict):
        """Appending a dict to the dataframe."""

        if item.get(FRAME, None) is None:
            item[FRAME] = Variable(FRAME, _Frame(self._len))
        else:
            item[FRAME].value.row_id = self._len

        # I believe its faster to loop through and check than list comp
        for key, val in item.items():
            assert isinstance(val, Variable)
            assert str(val.value.__class__.__name__) == self._schema.col_type(self._schema.col_index(key))

        for key, val in item.items():
            self._data_as_list[self._schema.col_index(key)].put(val)

        self._len += 1

    def seal(self):
        """
        Function to seal the dataframe. A sealed dataframe will not accept anymore data and
        the act of sealing the dataframe will trigger blocked iterators to pull the remaining data.
        """
        with self._data_lock:
            self._sealed = True

        # Release all blocked iters.
        with self._block_lock:
            for _, id_event in self._map_blocked.items():
                for _, event in id_event:
                    event.set()
            self._map_blocked.clear()
            for _, id_event in self._window_start_blocked.items():
                for _, event in id_event:
                    event.set()
            self._window_start_blocked.clear()
            for _, id_event in self._window_end_blocked.items():
                for _, event in id_event:
                    event.set()
            self._window_end_blocked.clear()

    def gc_data(self):
        """Internal garbage collection function to trigger towhee.Array GC."""
        with self._data_lock:
            vals = [value for _, value in self._iterators.items()]
            self._min_offset = min(vals) if len(vals) != 0 else -1
            for x in self._data_as_list:
                x.gc(self._min_offset)

    def gc_blocked(self):
        with self._block_lock:
            for key, val in list(self._map_blocked.items()):
                if len(val) == 0:
                    del self._map_blocked[key]
            for key, val in list(self._window_start_blocked.items()):
                if len(val) == 0:
                    del self._window_start_blocked[key]
            for key, val in list(self._window_end_blocked.items()):
                if len(val) == 0:
                    del self._window_end_blocked[key]

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

    def remove_iter(self, iter_id):
        """
        Removing the iterator from the df. Allows for more garbage collecting.
        Args:
            iter_id (`int`):
                The iterator's id.
        """
        with self._iterator_lock:
            if iter_id in self._iterators:
                del self._iterators[iter_id]
        self.gc_data()

    def ack(self, iter_id, offset):
        """
        An acknowledgement (ack) will notice the `DataFrame`s that the rows already
        iterated over are no longer used, and can be garbage collected. Up to this index
        but not including.

        Args:
            iter_id (`int`):
                The iterator id to set offset for.
            offset (`int`):
                The latest accepted offset.
        """
        with self._iterator_lock:
            if self._iterators[iter_id] <= (offset):
                self._iterators[iter_id] = offset
        # TODO: Figure out better algorithm for gc'ing
        self.gc_data()

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
        with self._block_lock:
            index = offset + count
            if self._map_blocked.get(index) is None:
                self._map_blocked[index] = []
            self._map_blocked[index].append((iter_id, event))

    def notify_window_block(self, iter_id, event, edge, cutoff):
        """
        Used by window based iterators to notify the dataframe that it is waiting on a value.

        Args:
            iter_id (`int`):
                The iterator id of the blocked iterator.
            event (`threading.Event`):
                The event to trigger once the value being waited on is available.
            edge ('start', 'end'):
                Waiting for a value >= than start, or waiting for a value greater then end.
            cutoff ("timestamp" | "row_id", `int`):
                The window cutoff that is is being waited on.
        """

        with self._block_lock:
            if edge == 'start':
                if self._window_start_blocked.get(cutoff) is None:
                    self._window_start_blocked[cutoff] = []
                self._window_start_blocked[cutoff].append((iter_id, event))

            elif edge == 'end':
                if self._window_end_blocked.get(cutoff) is None:
                    self._window_end_blocked[cutoff] = []
                self._window_end_blocked[cutoff].append((iter_id, event))

    def unblock_all(self):
        with self._block_lock:
            for _, id_event in self._map_blocked.items():
                for it_id, event in id_event:
                    del self._iterators[it_id]
                    event.set()

            self._map_blocked.clear()

            for _, id_event in self._window_start_blocked.items():
                for it_id, event in id_event:
                    del self._iterators[it_id]
                    event.set()

            self._window_start_blocked.clear()

            for _, id_event in self._window_end_blocked.items():
                for it_id, event in id_event:
                    del self._iterators[it_id]
                    event.set()

            self._window_end_blocked.clear()

    def unblock_iter(self, iter_id):
        """
        Forceful unblocking/killing of the blocked iter.
        """
        with self._block_lock:
            index = None
            for _, id_event in self._map_blocked.items():
                for i, (ids, event) in enumerate(id_event):
                    if ids == iter_id:
                        del self._iterators[iter_id]
                        event.set()
                        index = i
                        break
                if index is not None:
                    del id_event[index]
                    self.gc_blocked()
                    return

            for _, id_event in self._window_start_blocked.items():
                for i, (ids, event) in enumerate(id_event):
                    if ids == iter_id:
                        del self._iterators[iter_id]
                        event.set()
                        index = i
                        break
                if index is not None:
                    del id_event[index]
                    self.gc_blocked()
                    return

            for _, id_event in self._window_end_blocked.items():
                for i, (ids, event) in enumerate(id_event):
                    if ids == iter_id:
                        del self._iterators[iter_id]
                        event.set()
                        index = i
                        break
                if index is not None:
                    del id_event[index]
                    self.gc_blocked()
                    return


class Responses(Enum):
    """
    Response Codes between DF and Iterators
    """
    INDEX_GC = 1
    INDEX_OOB_UNSEALED = 2
    INDEX_OOB_SEALED = 3
    APPROVED_CONTINUE = 4
    APPROVED_DONE = 5
    KILLED = 6
    UNKOWN_ERROR = 7
    WINDOW_NOT_DONE = 8
    WINDOW_PASSED = 9
    FUTURE_WINDOW = 10
    OLD_WINDOW = 11
    FUTURE_WINDOW_SEALED = 12
    EMPTY = 13
    EMPTY_SEALED = 14
