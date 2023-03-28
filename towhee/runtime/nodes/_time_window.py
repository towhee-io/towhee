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


from towhee.runtime.constants import TimeWindowConst
from towhee.runtime.data_queue import Empty

from ._window import Window


class TimeWindow(Window):
    """Window operator

      inputs:       ---1--2-3-----4-5-6--->
      timestamp:    0 ------1000ms-------2000ms---->

      node definition:

        [    time_window('input', 'output', time_range_sec=1, time_step_sec=1, timestamp_col: 'timestamp', callable=lambda i: sum(i))     ]

      windows: [0s, 1s) [1s, 2s)
      every step:

        callable([1, 2, 3]) -> 6
        callable([4, 5, 6]) -> 15

      outputs:
        ----6-----15------->
    """
    def _init(self):
        self._time_range_sec = self._node_repr.iter_info.param[TimeWindowConst.param.time_range_sec]
        self._time_step_sec = self._node_repr.iter_info.param[TimeWindowConst.param.time_step_sec]
        self._timestamp_index = self._node_repr.iter_info.param[TimeWindowConst.param.timestamp_col]
        self._buffer = _TimeWindowBuffer(self._time_range_sec, self._time_step_sec)

    def _get_buffer(self):
        while True:
            data = self.input_que.get_dict()
            if data is None:
                # end of the data_queue
                if self._buffer is not None and self._buffer.data:
                    ret = self._buffer.data
                    self._buffer = self._buffer.next()
                    return self._to_cols(ret)
                self._set_finished()
                return None

            if not self.side_by_to_next(data):
                return None

            process_data = dict((key, data.get(key)) for key in self._node_repr.inputs if data.get(key) is not Empty())
            if not process_data or data[self._timestamp_index] is Empty():
                continue            

            if self._buffer(process_data, data[self._timestamp_index]) and self._buffer.data:
                ret = self._buffer.data
                self._buffer = self._buffer.next()
                return self._to_cols(ret)


class _TimeWindowBuffer:
    '''
    TimeWindow
    The unit of timestamp is milliseconds, the unit of window(range, step) is seconds.
    '''

    def __init__(self, time_range_sec: int, time_step_sec: int, start_time_sec: int = 0):
        self._start_time_m = start_time_sec * 1000
        self._end_time_m = self._start_time_m + time_range_sec * 1000
        self._next_start_time_m = (start_time_sec + time_step_sec) * 1000
        self._time_range_sec = time_range_sec
        self._time_step_sec = time_step_sec
        self._window = []
        self._next_window = None

    def __call__(self, row_data, timestamp: int) -> bool:
        if timestamp < self._start_time_m:
            return False

        if timestamp < self._end_time_m:
            self._window.append(row_data)
            if timestamp >= self._next_start_time_m:
                if self._next_window is None:
                    self._next_window = _TimeWindowBuffer(self._time_range_sec, self._time_step_sec, self._next_start_time_m // 1000)
                self._next_window(row_data, timestamp)
            return False

        if len(self._window) == 0:
            self._start_time_m = timestamp // 1000 // self._time_step_sec * self._time_step_sec * 1000
            self._end_time_m = self._start_time_m + self._time_range_sec * 1000
            self._next_start_time_m = (self._start_time_m // 1000 + self._time_step_sec) * 1000
            if self._start_time_m <= timestamp < self._end_time_m:
                self(row_data, timestamp)
            return False

        if self._next_window is None:
            self._next_window = _TimeWindowBuffer(self._time_range_sec, self._time_step_sec, self._next_start_time_m // 1000)
        self._next_window(row_data, timestamp)
        return True

    @property
    def data(self):
        return self._window

    def next(self):
        return self._next_window
