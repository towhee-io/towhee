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

from .node import Node


class Window(Node):
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

    def __init__(self, node_repr: 'NodeRepr',
                 op_pool: 'OperatorPool',
                 in_ques: List['DataQueue'],
                 out_ques: List['DataQueue']):

        super().__init__(node_repr, op_pool, in_ques, out_ques)
        self._init()

    def _init(self):
        self._size = self._node_repr.iter_info.param[WindowConst.param.size]
        self._step = self._node_repr.iter_info.param[WindowConst.param.step]
        self._cur_index = -1
        self._input_que = self._in_ques[0]
        self._schema = self._in_ques[0].schema
        self._buffer = _WindowBuffer(self._size, self._step)
        self._row_buffer = []

    def _get_buffer(self):
        while True:
            data = self._input_que.get()
            if data is None:
                # end of the data_queue
                if self._buffer is not None and self._buffer.data:
                    ret = self._buffer.data
                    self._buffer = self._buffer.next()
                    return self._to_cols(ret)
                else:
                    return None

            self._cur_index += 1
            self._row_buffer.append(data)
            if self._buffer(data, self._cur_index) and self._buffer.data:
                ret = self._buffer.data
                self._buffer = self._buffer.next()
                return self._to_cols(ret)

    def _to_cols(self, rows: List[List]):
        cols = [[] for _ in self._schema]
        for row in rows:
            for i in range(len(self._schema)):
                cols[i].append(row[i])
        ret = {}
        for i in range(len(self._schema)):
            ret[self._schema[i]] = cols[i]
        return ret

    def process_step(self) -> bool:
        """
        Process each window data.
        """
        in_buffer = self._get_buffer()
        if in_buffer is None:
            if self._row_buffer:
                cols = self._to_cols(self._row_buffer)
                for out_que in self._output_ques:
                    if not out_que.batch_put_dict(cols):
                        self._set_stopped()
                        return True
                self._row_buffer = []
            self._set_finished()
            return True

        process_data = [in_buffer.get(key) for key in self._node_repr.inputs]
        succ, outputs, msg = self._call(process_data)
        if not succ:
            self._set_failed(msg)
            return True

        size = len(self._node_repr.outputs)
        if size > 1:
            output_map = dict((self._node_repr.outputs[i], [outputs[i]])
                              for i in range(size))
        else:
            output_map = {}
            output_map[self._node_repr.outputs[0]] = [outputs]

        cols = self._to_cols(self._row_buffer)
        cols.update(output_map)

        for out_que in self._output_ques:
            if not out_que.batch_put_dict(cols):
                self._set_stopped()
                return True
        self._row_buffer = []
        return False


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
