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


from typing import List, Dict

from towhee.runtime.data_queue import Empty
from towhee.runtime.time_profiler import Event

from .node import Node
from ._single_input import SingleInputMixin


class WindowBase(Node, SingleInputMixin):
    """
    Window node.
    """

    def __init__(self, node_repr: 'NodeRepr',
                 op_pool: 'OperatorPool',
                 in_ques: List['DataQueue'],
                 out_ques: List['DataQueue'],
                 time_profiler: 'TimeProfiler'):

        super().__init__(node_repr, op_pool, in_ques, out_ques, time_profiler)
        self._init()

    def _init(self):
        raise NotImplementedError

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, buf):
        self._buffer = buf

    def _window_index(self, data):  # pylint: disable=unused-argument
        raise NotImplementedError

    def _get_buffer(self):
        while True:
            data = self.input_que.get_dict()
            if data is None:
                # end of the data_queue
                if self.buffer is not None and self.buffer.data:
                    ret = self.buffer.data
                    self.buffer = self.buffer.next()
                    return self._to_cols(ret)
                self._set_finished()
                return None

            if not self.side_by_to_next(data):
                return None

            process_data = dict((key, data.get(key)) for key in self._node_repr.inputs if data.get(key) is not Empty())

            if not process_data:
                continue

            index = self._window_index(data)
            if index < 0:
                continue

            if self.buffer(process_data, index) and self.buffer.data:
                ret = self.buffer.data
                self.buffer = self.buffer.next()
                return self._to_cols(ret)

    def _to_cols(self, rows: List[Dict]):
        ret = dict((key, []) for key in self._node_repr.inputs)
        for row in rows:
            for k, v in row.items():
                ret[k].append(v)
        return ret

    def process_step(self) -> bool:
        """
        Process each window data.
        """
        self._time_profiler.record(self.uid, Event.queue_in)
        in_buffer = self._get_buffer()
        if in_buffer is None:
            return

        process_data = [in_buffer.get(key) for key in self._node_repr.inputs]
        self._time_profiler.record(self.uid, Event.process_in)
        succ, outputs, msg = self._call(process_data)
        self._time_profiler.record(self.uid, Event.process_out)
        assert succ, msg

        size = len(self._node_repr.outputs)
        if size > 1:
            output_map = dict((self._node_repr.outputs[i], outputs[i])
                              for i in range(size))
        elif size == 1:
            output_map = {}
            output_map[self._node_repr.outputs[0]] = outputs
        else:
            output_map = {}

        self._time_profiler.record(self.uid, Event.queue_out)
        self.data_to_next(output_map)
