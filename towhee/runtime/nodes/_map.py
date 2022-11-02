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


from typing import Generator

from .node import Node
from towhee.runtime.data_queue import Empty


class Map(Node):
    """Map operator.

        Project each element of an input sequence into a new form.
        two cases:
           1. The operator return normal element.

               ---1---2---3---4--->
           [   map('input', 'output', lambda i: i + 1)    ]
               ---2---4---6---8--->

           2. The operator return a generator.

           def func(i):
                num = 0
                while num < i:
                    yield num
                    num += 1

               ---1---2---3---4--->
           [   map('input', 'output', func)    ]
               ---[0]---[0, 1]---[0, 1, 2]---[0, 1, 2, 3]--->
    """

    def process_step(self) -> bool:
        """
        Called for each element.
        """

        data = self._in_ques[0].get_dict()
        if data is None:
            self._set_finished()
            return True

        process_data = [data.get(key) for key in self._node_repr.inputs]
        if Empty() not in process_data:
            succ, outputs, msg = self._call(process_data)
            if not succ:
                self._set_failed(msg)
                return True
            if isinstance(outputs, Generator):
                outputs = self._get_from_generator(outputs, len(self._node_repr.outputs))

            size = len(self._node_repr.outputs)
            if size > 1:
                output_map = dict((self._node_repr.outputs[i], outputs[i])
                                for i in range(size))
            elif size == 0:
                output_map = {}
            else:
                output_map = {}
                output_map[self._node_repr.outputs[0]] = outputs

            data.update(output_map)

        for out_que in self._output_ques:
            if not out_que.put_dict(data):
                self._set_stopped()
                return True

        return False

    def _get_from_generator(self, gen, size):
        if size == 1:
            return list(gen)
        ret = [ [] for _ in range(size)]
        for data in gen:
            for i in range(size):
                ret[i].append(data[i])
        return ret
