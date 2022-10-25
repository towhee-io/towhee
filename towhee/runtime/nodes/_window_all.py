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

from .node import Node



class WindowAll(Node):
    """Window operator

      inputs: ---1-2-3-4-5-6--->

      node definition:

        [    window_all('input', 'output', callable=lambda i: sum(i))     ]

      step:

        callable([1, 2, 3, 4, 5, 6]) -> 21

      outputs:
        ----21---->
    """
    def __init__(self, node_info: 'NodeInfo',
                 op_pool: 'OperatorPool',
                 in_ques: List['DataQueue'],
                 out_ques: List['DataQueue']):

        super().__init__(node_info, op_pool, in_ques, out_ques)
        self._input_que = in_ques[0]
        self._schema = in_ques[0].schema

    def _get_buffer(self):
        data = self._input_que.get()
        if not data:
            return None
        else:
            assert len(self._schema) == len(data)
            cols = [[i] for i in data]

        while True:
            data = self._input_que.get()
            if data is None:
                break

            for i in range(len(self._schema)):
                if data[i] is not Empty():
                    cols[i].append(data[i])

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

        in_buffer.update(output_map)

        for out_que in self._output_ques:
            if not out_que.batch_put_dict(in_buffer):
                self._set_stopped()
                return True
        self._set_finished()
        return True
