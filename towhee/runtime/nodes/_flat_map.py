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
from typing import List, Any

from towhee.runtime.data_queue import Empty
from towhee.runtime.time_profiler import Event

from .node import Node
from .single_input import SingleInputMixin


class FlatMap(Node, SingleInputMixin):
    """
    FlatMap Operator.

    FlatMap transforms the iterable/nested outputs into one or more elements, i.e. split elements, unnest iterables.

    i.e.
            ---[0, 1, 2, 3]--->
        [    FlatMap('input', 'output', lambda i: i)    ]
            ---0---1---2---3--->
    """
    def process_step(self) -> List[Any]:
        self._time_profiler.record(self.uid, Event.queue_in)
        data = self.read_row()
        if data is None or not self.side_by_to_next(data):
            return None
        process_data = [data.get(key) for key in self._node_repr.inputs]

        if any((item is Empty() for item in process_data)):
            return None

        self._time_profiler.record(self.uid, Event.process_in)        
        succ, outputs, msg = self._call(process_data)
        assert succ, msg

        size = len(self._node_repr.outputs)
        for output in outputs:
            if size > 1:
                output_map = {self._node_repr.outputs[i]: output[i] for i in range(size)}
            else:
                output_map = {self._node_repr.outputs[0]: output}
            if not self.data_to_next(output_map):
                return None

        self._time_profiler.record(self.uid, Event.process_out)
        self._time_profiler.record(self.uid, Event.queue_out)
