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


from ._map import Map
from .node_info import NodeInfo
from .node import NodeStatus
from towhee.utils.log import engine_log


def create_node(node_info, op_pool, inputs, outputs):
    info = NodeInfo(node_info)
    if info.type == 'map':
        assert len(inputs) == 1
        return Map(info, op_pool, inputs, outputs)
    else:
        engine_log.error('Unkown node type: %s' % str(info.type))
        return None

all = [
    'NodeInfo',
    'NodeStatus',
    'create_node'
]
