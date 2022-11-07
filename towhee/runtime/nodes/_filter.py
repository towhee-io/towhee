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
from .node import Node


class Filter(Node):
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
                 out_ques: List['DataQueue']):
        super().__init__(node_repr, op_pool, in_ques, out_ques)
        self._key_map = dict(zip(self._node_repr.outputs, self._node_repr.inputs))
        self._same_keys = list(set(self._key_map.keys()) & set(self._key_map.values()))

    def process_step(self) -> bool:
        assert len(self._node_repr.outputs) == len(self._node_repr.inputs)

        data = self._in_ques[0].get_dict()
        if data is None:
            self._set_finished()
            return True

        process_data = [data.get(key) for key in self._node_repr.iter_info.param[FilterConst.param.filter_by]]
        if Empty() not in process_data:
            succ, outputs, msg = self._call(process_data)
            if not succ:
                self._set_failed(msg)
                return True

            if outputs:
                output_map = {new_key: data[old_key] for new_key, old_key in self._key_map.items()}
                data.update(output_map)
            else:
                for k in self._same_keys:
                    del data[k]

        for out_que in self._output_ques:
            if not out_que.put_dict(data):
                self._set_stopped()
                return True

        return False
