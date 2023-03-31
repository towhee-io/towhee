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

from towhee.runtime.constants import FilterConst
from towhee.runtime.data_queue import Empty
from towhee.runtime.time_profiler import Event

from .node import Node
from ._single_input import SingleInputMixin


class Filter(Node, SingleInputMixin):
    """
    Filter Operator.

    Filter the input columns based on the selected filter_columns and filter.

    i.e.
            ---1---2---3---4--->
        [   filter('input', 'output', lambda i: i > 2)    ]
            ---3---4--->
    """
    def __init__(self, node_repr: 'NodeRepr',
                 op_pool: 'OperatorPool',
                 in_ques: List['DataQueue'],
                 out_ques: List['DataQueue'],
                 time_profiler: 'TimeProfiler'):
        super().__init__(node_repr, op_pool, in_ques, out_ques, time_profiler)
        self._key_map = dict(zip(self._node_repr.outputs, self._node_repr.inputs))

    def process_step(self):
        self._time_profiler.record(self.uid, Event.queue_in)
        data = self.read_row()
        if data is None or not self.side_by_to_next(data):
            return None

        process_data = [data.get(key) for key in self._node_repr.iter_info.param[FilterConst.param.filter_by]]
        if any((i is Empty() for i in process_data)):
            return None

        self._time_profiler.record(self.uid, Event.process_in)
        succ, is_need, msg = self._call(process_data)
        self._time_profiler.record(self.uid, Event.process_out)
        assert succ, msg
        self._time_profiler.record(self.uid, Event.queue_out)
        if is_need:
            output_map = {new_key: data[old_key] for new_key, old_key in self._key_map.items()}
            self.data_to_next(output_map)


