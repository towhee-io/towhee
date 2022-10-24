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
from ._window import Window
from ._time_window import TimeWindow
from ._window_all import WindowAll
from ._concat import Concat
from ._filter import Filter
from ._flat_map import FlatMap
from .node import NodeStatus
import towhee.runtime.constants as constants
from towhee.utils.log import engine_log


def create_node(node_repr, op_pool, inputs, outputs):
    if node_repr.iter_info.type == constants.MapConst.name:
        assert len(inputs) == 1
        return Map(node_repr, op_pool, inputs, outputs)
    elif node_repr.iter_info.type == constants.WindowConst.name:
        assert len(inputs) == 1
        return Window(node_repr, op_pool, inputs, outputs)
    if node_repr.iter_info.type == constants.FilterConst.name:
        assert len(inputs) == 1
        return Filter(node_repr, op_pool, inputs, outputs)
    if node_repr.iter_info.type == constants.TimeWindowConst.name:
        assert len(inputs) == 1
        return TimeWindow(node_repr, op_pool, inputs, outputs)
    if node_repr.iter_info.type == constants.WindowAllConst.name:
        assert len(inputs) == 1
        return WindowAll(node_repr, op_pool, inputs, outputs)
    if node_repr.iter_info.type == constants.ConcatConst.name:
        return Concat(node_repr, op_pool, inputs, outputs)
    if node_repr.iter_info.type == constants.FlatMapConst.name:
        assert len(inputs) == 1
        return FlatMap(node_repr, op_pool, inputs, outputs)
    else:
        engine_log.error('Unknown node iteration type: %s', str(node_repr.iter_info.type))
        return None


__all__ = [
    'NodeStatus',
    'create_node'
]
