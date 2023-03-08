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

from .node import Node
from towhee.runtime.data_queue import Empty
from towhee.runtime.performance_profiler import Event


class FlatMap(Node):
    """
    FlatMap Operator.

    FlatMap transforms the iterable/nested outputs into one or more elements, i.e. split elements, unnest iterables.

    i.e.
            ---[0, 1, 2, 3]--->
        [    FlatMap('input', 'output', lambda i: i)    ]
            ---0---1---2---3--->
    """
    def __init__(self, node_repr, op_pool, in_ques, out_ques, time_profiler):
        super().__init__(node_repr, op_pool, in_ques, out_ques, time_profiler)
        self._input_q = self._in_ques[0]
        self._side_by_keys = list(set(self._input_q.schema) - set(self._node_repr.outputs))

    def process_step(self) -> List[Any]:
        self._time_profiler.record(self.uid, Event.queue_in)
        data = self._in_ques[0].get_dict()
        if data is None:
            self._set_finished()
            return True

        side_by = dict((k, data[k]) for k in self._side_by_keys)
        process_data = [data.get(key) for key in self._node_repr.inputs]
        if any((i is Empty() for i in process_data)):
            for out_que in self._output_ques:
                if not out_que.put_dict(side_by):
                    self._set_stopped()
                    return True
        else:
            self._time_profiler.record(self.uid, Event.process_in)
            succ, outputs, msg = self._call(process_data)
            if not succ:
                self._set_failed(msg)
                return True

            size = len(self._node_repr.outputs)

            for output in outputs:
                if size > 1:
                    output_map = {self._node_repr.outputs[i]: output[i] for i in range(size)}
                else:
                    output_map = {self._node_repr.outputs[0]: output}

                side_by.update(output_map)

                for out_que in self._output_ques:
                    if not out_que.put_dict(side_by):
                        self._set_stopped()
                        return True

                side_by = {}
            self._time_profiler.record(self.uid, Event.process_out)
            self._time_profiler.record(self.uid, Event.queue_out)

        return False
