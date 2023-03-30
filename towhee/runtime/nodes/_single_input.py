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


class SingleInputMixin:
    """
    For single input node.
    """

    @property
    def input_que(self):
        return self._in_ques[0]

    @property
    def side_by_cols(self):
        if not hasattr(self, '_side_by_cols'):
            self._side_by_cols = list(set(self.input_que.schema) - set(self._node_repr.outputs))
        return self._side_by_cols

    def read_row(self):
        data = self.input_que.get_dict()
        if data is None:
            self._set_finished()
            return None
        return data

    def side_by_to_next(self, data):
        side_by = dict((k, data[k]) for k in self.side_by_cols)
        return self.data_to_next(side_by)
