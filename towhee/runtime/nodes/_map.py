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
import traceback

from .node import Node, NodeStatus
from towhee.utils.log import engine_log


class Map(Node):

    def process_step(self) -> bool:
        '''
        If return True, means finished.
        '''
        datas = self._in_ques[0].get_dict()
        if datas is None:
            self._set_finished()
            return True

        process_data = [datas.get(key) for key in self._node_info.input_schema]
        succ, outputs, msg = self._call(process_data)
        if not succ:
            self._set_failed(msg)
            return True

        output_map = dict((self._node_info.output_schema[i], outputs[i]) for i in range(len(self._node_info.output_schema)))
        datas.update(output_map)
        for out_que in self._output_ques:
            if not out_que.put_dict(datas):
                self._set_stopped()
                return True

        return False
