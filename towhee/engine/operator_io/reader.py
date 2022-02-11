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

from abc import ABC, abstractmethod
import threading
from collections import namedtuple
from typing import Dict, Tuple, Union, List, Optional

from towhee.dataframe import DataFrame, Variable, DataFrameIterator


class ReaderBase(ABC):
    """
    The reader base class.
    The read() could be blocking or non-blocking function, if it's a blocking function,
    the runner may be blocked. When need to stop the graph, we call close to interrupting it.
    """

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class DataFrameReader(ReaderBase):
    """
    Read data from input dataframes, unpack and combine data.
    One op_ctx has one dataframe reader.
    """

    def __init__(self, it: DataFrameIterator, op_inputs_index: Dict[str, int]):
        self._op_inputs_index = op_inputs_index
        self._iter = it

    @abstractmethod
    def read(self) -> Union[Dict[str, any], List[Dict[str, any]]]:
        pass

    @property
    def size(self) -> int:
        return self._iter.accessible_size

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def _to_op_inputs(self, cols: Tuple[Variable]) -> Dict[str, any]:
        """
        Read from cols, combine op inputs
        """
        ret = {}
        for key, index in self._op_inputs_index.items():
            ret[key] = cols[index].value
        return ret


class BlockMapReaderWithOriginData(DataFrameReader):
    """
    Return both op's input data and origin data.
    """

    def __init__(
        self,
        input_df: DataFrame,
        op_inputs_index: Dict[str, int]
    ):
        super().__init__(input_df.map_iter(True), op_inputs_index)
        self._lock = threading.Lock()
        self._close = False

    def read(self) -> Tuple[Dict[str, any], Tuple]:
        """
        Read data from dataframe, get cols by operator_repr info
        """
        if self._close:
            raise StopIteration

        with self._lock:
            data = next(self._iter)
            if self._close:
                raise StopIteration

            if not data:
                return {}, ()
            return self._to_op_inputs(data), data

    def close(self):
        self._close = True
        self._iter.notify()


class BatchFrameReader(DataFrameReader):
    """
    Batch reader.
    """

    def __init__(self, input_df: DataFrame, op_inputs_index: Dict[str, int],
                 batch_size: int, step: int):
        assert batch_size >= 1 and step >= 1
        super().__init__(input_df.batch_iter(batch_size, step, True), op_inputs_index)
        self._close = False
        self._lock = threading.Lock()

    def read(self) -> List[Dict[str, any]]:
        if self._close:
            raise StopIteration

        with self._lock:
            data = next(self._iter)
            if self._close:
                raise StopIteration

            if not data:
                return []
            else:
                res = []
                for row in data:
                    data_dict = self._to_op_inputs(row)
                    res.append(namedtuple('input', data_dict.keys())(**data_dict))
                return res

    def close(self):
        self._close = True
        self._iter.notify()


class _TimeWindow:
    '''
    '''

    def __init__(self, time_range_sec: int, time_step_sec: int, start_time_sec: int = 0):
        self._start_time_m = start_time_sec * 1000
        self._end_time_m = self._start_time_m + time_range_sec * 1000
        self._next_start_time_m = (start_time_sec + time_step_sec) * 1000
        self._time_range_sec = time_range_sec
        self._time_step_sec = time_step_sec
        self._window = []
        self._next_window = None

    def __call__(self, row_data) -> bool:
        frame = row_data[-1].value
        if frame.timestamp < self._start_time_m:
            return False

        if frame.timestamp < self._end_time_m:
            self._window.append(row_data)
            if frame.timestamp >= self._next_start_time_m:
                if self._next_window is None:
                    self._next_window = _TimeWindow(self._time_range_sec, self._time_step_sec, self._next_start_time_m // 1000)
                self._next_window(row_data)
            return False

        if len(self._window) == 0:
            self._start_time_m = frame.timestamp // 1000 // self._time_step_sec * self._time_step_sec * 1000
            self._end_time_m = self._start_time_m + self._time_range_sec * 1000
            self._next_start_time_m = (self._start_time_m // 1000 + self._time_step_sec) * 1000
            if frame.timestamp >= self._start_time_m and frame.timestamp < self._end_time_m:
                self(row_data)
            return False

        if self._next_window is None:
            self._next_window = _TimeWindow(self._time_range_sec, self._time_step_sec, self._next_start_time_m // 1000)
        self._next_window(row_data)
        return True

    @property
    def data(self):
        return self._window

    @property
    def next_window(self):
        return self._next_window


class TimeWindowReader(DataFrameReader):
    """
    Time window reader
    """

    def __init__(
        self,
        input_df: DataFrame,
        op_inputs_index: Dict[str, int],
        time_range_sec: int,
        time_step_sec: int
    ):
        super().__init__(input_df.map_iter(True), op_inputs_index)
        self._window = _TimeWindow(time_range_sec, time_step_sec)
        self._lock = threading.Lock()
        self._close = False

    def read(self) -> Tuple[Dict[str, any], Tuple]:
        """
        Read data from dataframe, get cols by operator_repr info
        """
        with self._lock:
            if self._close or self._window is None:
                raise StopIteration

            while True:
                if self._window is None:
                    raise StopIteration

                try:
                    data = next(self._iter)
                except StopIteration:
                    if self._window is not None:
                        ret = [self._to_op_inputs(row) for row in self._window.data]
                        self._window = self._window.next_window
                        return ret

                if self._close:
                    raise StopIteration

                if data is None:
                    continue

                is_end = self._window(data)
                if is_end and len(self._window.data) != 0:
                    ret = [self._to_op_inputs(row) for row in self._window.data]
                    self._window = self._window.next_window
                    return ret

    def close(self):
        self._close = True
        self._iter.notify()
