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

from towhee.runtime.data_queue import Empty

from towhee.runtime.performance_profiler import Event

from .node import Node
from .single_input import SingleInputMixin


class WindowAll(Node, SingleInputMixin):
    """Window operator

      inputs: ---1-2-3-4-5-6--->

      node definition:

        [    window_all('input', 'output', callable=lambda i: sum(i))     ]

      step:

        callable([1, 2, 3, 4, 5, 6]) -> 21

      outputs:
        ----21---->
    """
    def _get_buffer(self):
        ret = dict((key, []) for key in self._node_repr.inputs)
        while True:
            data = self.input_que.get_dict()
            if data is None:
                return ret

            if not self.side_by_to_next(data):
                return None

            for key in self._node_repr.inputs:
                if data.get(key) is not Empty():
                    ret[key].append(data.get(key))

    def process_step(self) -> bool:
        """
        Process each window data.
        """
        self._time_profiler.record(self.uid, Event.queue_in)
        in_buffer = self._get_buffer()
        if in_buffer is None:
            return

        if all([False if col else True for col in in_buffer.values()]):
            self._set_finished()
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
        if not self.data_to_next(output_map):
            return
        self._set_finished()

