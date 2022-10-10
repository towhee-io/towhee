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


class NodeInfo:
    '''
    Node info.
    TODO: use node_repr.
    '''

    KEYS = ['name', 'type', 'op_info', 'config', 'input_schema', 'output_schema']

    def __init__(self, data):
        self._data = data
        self.op_info = OpInfo(data.get('op_info', {}))

    def __getattr__(self, name: str):
        if name not in self.KEYS:
            return None
        elif name == 'op_info':
            return self._meta

        return self._data.get(name)


class OpInfo:
    '''
    OpInfo
    TODO: use node_repr
    '''

    KEYS = ['hub_id', 'name', 'args', 'kwargs', 'tag']

    def __init__(self, info):
        self._data = info

    def __getattr__(self, name: str):
        if name not in self.KEYS:
            return None

        return self._data.get(name)

    def __str__(self):
        return str(self._data)
