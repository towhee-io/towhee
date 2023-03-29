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


from towhee.runtime.constants import (
    MapConst,
    WindowAllConst,
    WindowConst,
    ReduceConst,
    FilterConst,
    TimeWindowConst,
    FlatMapConst,
    ConcatConst,
    OutputConst
)
from towhee.utils.log import engine_log

from ._map import Map
from ._window import Window
from ._time_window import TimeWindow
from ._window_all import WindowAll
from ._reduce import Reduce
from ._concat import Concat
from ._filter import Filter
from ._flat_map import FlatMap
from ._output import Output
from .node import NodeStatus


def create_node(node_repr, op_pool, inputs, outputs, time_profiler=None):
    if node_repr.uid == OutputConst.name:
        assert len(inputs) == 1
        return Output(node_repr, op_pool, inputs, outputs, time_profiler)
    if node_repr.iter_info.type == MapConst.name:
        assert len(inputs) == 1
        return Map(node_repr, op_pool, inputs, outputs, time_profiler)
    elif node_repr.iter_info.type == WindowConst.name:
        assert len(inputs) == 1
        return Window(node_repr, op_pool, inputs, outputs, time_profiler)
    if node_repr.iter_info.type == FilterConst.name:
        assert len(inputs) == 1
        return Filter(node_repr, op_pool, inputs, outputs, time_profiler)
    if node_repr.iter_info.type == TimeWindowConst.name:
        assert len(inputs) == 1
        return TimeWindow(node_repr, op_pool, inputs, outputs, time_profiler)
    if node_repr.iter_info.type == WindowAllConst.name:
        assert len(inputs) == 1
        return WindowAll(node_repr, op_pool, inputs, outputs, time_profiler)
    if node_repr.iter_info.type == ReduceConst.name:
        assert len(inputs) == 1
        return Reduce(node_repr, op_pool, inputs, outputs, time_profiler)
    if node_repr.iter_info.type == ConcatConst.name:
        return Concat(node_repr, op_pool, inputs, outputs, time_profiler)
    if node_repr.iter_info.type == FlatMapConst.name:
        assert len(inputs) == 1
        return FlatMap(node_repr, op_pool, inputs, outputs, time_profiler)
    else:
        engine_log.error('Unknown node iteration type: %s', str(node_repr.iter_info.type))
        return None


__all__ = [
    'NodeStatus',
    'create_node'
]
