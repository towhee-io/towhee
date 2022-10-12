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
from .node import NodeStatus
from towhee.runtime.node_repr import NodeRepr
from towhee.utils.log import engine_log


def create_node(node_info, op_pool, inputs, outputs):
    node_repr = NodeRepr.from_dict(node_info)
    if node_repr.iter_info.type == 'map':
        assert len(inputs) == 1
        return Map(node_repr, op_pool, inputs, outputs)
    else:
        engine_log.error('Unknown node iteration type: %s', str(info.iter_info.type))
        return None


__all__ = [
    'NodeStatus',
    'create_node'
]
