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


from typing import List

from towhee.runtime.constants import WindowConst

from ._window_base import WindowBase



class Window(WindowBase):
    """Window operator

      inputs: ---1-2-3-4-5-6--->

      node definition:

        [    window('input', 'output', size=3, step=2, callable=lambda i: sum(i))     ]

      every step:

        callable([1, 2, 3]) -> 6
        callable([3, 4, 5]) -> 12
        callable([5, 6]) -> 11

      outputs:
        ----6-12-11---->
    """

    def _init(self):
        self._size = self._node_repr.iter_info.param[WindowConst.param.size]
        self._step = self._node_repr.iter_info.param[WindowConst.param.step]
        self._cur_index = -1
        self._buffer = _WindowBuffer(self._size, self._step)

    def _window_index(self, data):  # pylint: disable=unused-argument
        self._cur_index += 1
        return self._cur_index


class _WindowBuffer:
    '''
    Collect data by size and step.
    '''
    def __init__(self, size: int, step: int, start_index: int = 0):
        self._start_index = start_index
        self._end_index = start_index + size
        self._next_start_index = start_index + step
        self._size = size
        self._step = step
        self._buffer = []
        self._next_buffer = None

    def __call__(self, data: List, index: int) -> bool:
        if index < self._start_index:
            return False

        if index < self._end_index:
            self._buffer.append(data)
            if index >= self._next_start_index:
                if self._next_buffer is None:
                    self._next_buffer = _WindowBuffer(self._size, self._step, self._next_start_index)
                self._next_buffer(data, index)
            return False

        if self._next_buffer is None:
            self._next_buffer = _WindowBuffer(self._size, self._step, self._next_start_index)

        self._next_buffer(data, index)
        return True

    @property
    def data(self):
        return self._buffer

    def next(self):
        return self._next_buffer
