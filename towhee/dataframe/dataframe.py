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
from towhee.types._frame import _Frame, FRAME
# from towhee.types import equivalents


class DataFrame:
    """
    A `DataFrame` is a collection of immutable, potentixally heterogeneous blogs of data.

    Args:
        columns (`list[Tuple[str, Any]]`)
            The list of the column names and their corresponding data types.
        name (`str`):
            Name of the dataframe; `DataFrame` names should be the same as its
            representation.
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
        frame = _Frame()
        self._schema.add_col(FRAME, self._class_type(frame))
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
                for x in range(self._min_offset, self._min_offset + self.__len__()):
                    values = []
                    for i in range(len(self._data_as_list)):
                        val = self._data_as_list[i][x]
                        if isinstance(val, _Frame):
                            val = (val.row_id, val.prev_id, val.timestamp,)
                        values.append(str(val))
                    ret += formater.format(*values) + '\n'

            return ret

    def __len__(self):
        with self._data_lock:
            return self._len - self._min_offset

    @property
    def current_size(self):
        return self.__len__()

    @property
    def size(self):
        return self._len

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

    def get_window(self, start: int, end: int, step: int, comparator: str, iter_id: int = False):
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
            if iter_id is not False and self._iterators.get(iter_id) is None:
                return Responses.KILLED, None, None

        base_offset = self._iterators[iter_id]
        rets = []
        min_offset = None
        max_offset = base_offset
        next_offset = None
        next_window = start + step

        with self._data_lock:
            #  If the df is empty, we dont want to do anything but wait for a next value if unsealed.
            if self.__len__() == 0:
                if self.sealed:
                    return self._ret(Responses.EMPTY_SEALED, None, None, iter_id)
                else:
                    return self._ret(Responses.EMPTY, None, None, iter_id)
            #  If the first element in the df is larger than the window end we want to move to the next viable window
            if getattr(self._data_as_list[-1][base_offset], comparator) >= end:
                goal = getattr(self._data_as_list[-1][base_offset], comparator)
                new_start = start + step * ((goal - start)//step)
                new_end = (new_start - start) + end
                if new_end <= goal:
                    new_start += step
                    new_end += step
                return self._ret(Responses.OLD_WINDOW, None, (new_start, (new_start - start) + end), iter_id)

            #  If the last element of the df is smaller than the start of the window, proceed to next windows.
            elif getattr(self._data_as_list[-1][-1], comparator) < start:
                if self.sealed:
                    return self._ret(Responses.FUTURE_WINDOW_SEALED, None, None, iter_id)
                else:
                    return self._ret(Responses.FUTURE_WINDOW, None, self._len, iter_id)

            # iterating through values to see which ones fall in the window. min offset = first value, max offset = last value
            for i in range(base_offset, self._len):
                if getattr(self._data_as_list[-1][i], comparator) >= start and getattr(self._data_as_list[-1][i], comparator) < end:
                    if min_offset is None:
                        min_offset = i
                    if next_offset is None:
                        if getattr(self._data_as_list[-1][i], comparator) >= next_window:
                            next_offset = i
                    max_offset = i
                    rets.append(self.__getitem__(max_offset))
                elif getattr(self._data_as_list[-1][i], comparator) >= end:
                    if next_offset is None:
                        next_offset = i
                    break

            # If the max offset is the length of the df
            if max_offset == self._len - 1 and self.sealed:
                # Need to check if there is a next window that fits, if so, continue
                if next_offset:
                    return self._ret(Responses.APPROVED_CONTINUE, rets, next_offset, iter_id)
                # If not, return whats left and have iterator close.
                else:
                    return self._ret(Responses.APPROVED_DONE, rets, max_offset, iter_id)
            # If the last value that fits in window is also the last value of df, cant close
            # window yet as we dont know if the next value fits in the same window.
            elif max_offset == self._len - 1 and not self.sealed:
                return self._ret(Responses.WINDOW_NOT_DONE, None, None, iter_id)
            else:
                return self._ret(Responses.APPROVED_CONTINUE, rets, next_offset, iter_id)


    def get(self, offset: int, count: int = 1, iter_id: int = False):
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
            if iter_id is not False and self._iterators.get(iter_id) is None:
                return self._ret(Responses.KILLED, None, None, iter_id)

        with self._data_lock:
            # Up to iterators to decide what to do if the value they want is GCed.
            if offset < self._min_offset:
                return self._ret(Responses.INDEX_GC, None, None, iter_id)

            # The batch of data is available and there is more available.
            elif offset + count <= self._len:
                return self._ret(Responses.APPROVED_CONTINUE, [self.__getitem__(x) for x in range(offset, offset + count)], None, iter_id)

            elif self._sealed:
                # If no more data is left and trying to get more, iterator will stop and no data returned.
                if offset >= self._len:
                    return self._ret(Responses.INDEX_OOB_SEALED, None, None, iter_id)

                # If dataframe is sealed, remaining values will be returned and allow for next step
                else:
                    return self._ret(Responses.APPROVED_CONTINUE, [self.__getitem__(x) for x in range(offset, self._len)], None, iter_id)

            # If the requested offset is out of bounds but dataframe is still writing data.
            elif offset + count > self._len:
                return self._ret(Responses.INDEX_OOB_UNSEALED, None, None, iter_id)

            # Something broke.
            else:
                return self._ret(Responses.UNKOWN_ERROR, None, None, iter_id)

    def _ret(self, code, ret, offset, provide_code):
        if  provide_code is not False:
            return code, ret, offset
        else:
            return ret


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
                    if self._put_dict(x) == -1:
                        return
                elif isinstance(x, tuple):
                    if self._put_tuple(x) == -1:
                        return
                else:
                    raise ValueError('Input data is of wrong format.')

            frame = self._data_as_list[-1][-1]
            cur_len = self._len

        # Release blocked iterators if their criteria met.
        with self._block_lock:
            if len(self._map_blocked) > 0:
                ret = []
                for i, id_events in self._map_blocked.items():
                    if i <= cur_len:
                        for _, event in id_events:
                            event.set()
                        ret.append(i)
                for key in ret:
                    del self._map_blocked[key]

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
        if not isinstance(item[-1], _Frame):
            item.append(_Frame(row_id=self._len))
        else:
            if item[-1].empty:
                return -1
            item[-1].row_id = self._len

        # TODO: Figure out the situation on type checking
        # for i, x in enumerate(item):
            # print(equivalents.get(self._class_type(x), self._class_type(x)) , self._schema.col_type(i), flush=True)
            # assert equivalents.get(self._class_type(x), self._class_type(x)) == self._schema.col_type(i)

        for i, x in enumerate(item):
            self._data_as_list[i].put(x)

        self._len += 1

    def _put_dict(self, item: dict):
        """Appending a dict to the dataframe."""
        if item.get(FRAME, None) is None:
            item[FRAME] = _Frame(self._len)
        else:
            if item[FRAME].empty:
                return -1
            item[FRAME].row_id = self._len

        # I believe its faster to loop through and check than list comp
        # TODO: Add type checking
        # for key, val in item.items():
            # assert equivalents.get(self._class_type(val), self._class_type(val)) == self._schema.col_type(self._schema.col_index(key))

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

    def gc(self):
        self.gc_data()
        self.gc_blocked()

    def gc_data(self):
        """Garbage collection function to trigger towhee.Array GC."""
        with self._data_lock:
            vals = [value for _, value in self._iterators.items()]
            self._min_offset = min(vals + [self._len])
            for x in self._data_as_list:
                x.gc(self._min_offset)

    def gc_blocked(self):
        """Garbage collection for blocked iterator list."""
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

    def remove_iter(self, iter_id: int):
        """
        Removing the iterator from the df. Allows for more garbage collecting.
        Args:
            iter_id (`int`):
                The iterator's id.
        """
        self.unblock_iter(iter_id)
        try:
            with self._iterator_lock:
                if iter_id in self._iterators:
                    del self._iterators[iter_id]
        except KeyError:
            pass

        # self.gc_data()

    def ack(self, offset: int, iter_id: int):
        """
        An acknowledgement (ack) will notice the `DataFrame`s that the rows already
        iterated over are no longer used, and can be garbage collected. Up to this index
        but not including.

        Args:
            offset (`int`):
                The latest accepted offset.
            iter_id (`int`):
                The iterator id to set offset for.
        """
        with self._iterator_lock:
            if self._iterators[iter_id] <= (offset):
                self._iterators[iter_id] = offset
        # TODO: Figure out better algorithm for gc'ing
        # self.gc_data()

    def notify_map_block(self, event: threading.Event, offset: int, count: int, iter_id: int):
        """
        Used by map based iterators to notify the dataframe that it is waiting on a value.

        Args:
            event (`threading.Event`):
                The event to trigger once the value being waited on is available.
            offset (`int`):
                The offset that is being waited on.
            count (`int):
                How many values are being waited on.
            iter_id (`int`):
                The iterator id of the blocked iterator.
        """
        with self._block_lock:
            index = offset + count
            if self._map_blocked.get(index) is None:
                self._map_blocked[index] = []
            self._map_blocked[index].append((iter_id, event))

    def notify_window_block(self, event: threading.Event, edge: str, cutoff: Tuple[str, int], iter_id: int):
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
        """Forceful unblocking/killing of all blocked iter."""
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

    def unblock_iter(self, iter_id: int):
        """
        Forceful unblocking/killing of the selected iter.

        Args:
            iter_id (`int`):
                The id of the iterator being unblocked.
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
                    # self.gc_blocked()
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
                    # self.gc_blocked()
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
                    # self.gc_blocked()
                    return

    def _class_type(self, o):
        """Iternal function to find the full module path of object."""
        module = o.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return o.__class__.__name__
        return module + '.' + o.__class__.__name__


class Responses(Enum):
    """Response Codes between DF and Iterators."""
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
