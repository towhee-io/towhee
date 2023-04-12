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
import re
from typing import Dict, Any, List

from towhee.utils.log import engine_log


class NodeVisualizer:
    """
    Visualize the data of a node.
    """
    def __init__(self, name, data: Dict[str, Any]):
        self._name = name
        self._in = data['in']
        self._out = data['out']
        self._previous = data.get('previous')
        self._next = data.get('next')
        self._op_input = data['op_input']

    @property
    def name(self):
        return self._name

    @property
    def inputs(self):
        return self._in

    @property
    def outputs(self):
        return self._out

    @property
    def previous_node(self):
        return self._previous

    @property
    def next_node(self):
        return self._next

    @property
    def op_input(self):
        return self._op_input

    def _show_base(self):
        print('Node:', self._name)
        print('Previous Node:', self._previous)
        print('Next Node:', self._next)
        print('Op_Input:', self._op_input)

    def _show_inputs(self, tablefmt):
        if self._previous:
            for idx, i in enumerate(self._in):
                print('Input from', self._previous[idx] + ':')
                i.show(tablefmt=tablefmt)

    def _show_outputs(self, tablefmt):
        if self._next:
            for idx, i in enumerate(self._out):
                print('Output to', self._next[idx] + ':')
                i.show(tablefmt=tablefmt)

    def show_inputs(self, tablefmt=None):
        self._show_base()
        self._show_inputs(tablefmt)

    def show_outputs(self, tablefmt=None):
        self._show_base()
        self._show_outputs(tablefmt)

    def show(self, tablefmt=None):
        self._show_base()
        self._show_inputs(tablefmt)
        self._show_outputs(tablefmt)


class PipeVisualizer:
    """
    Visualize the data from single pipeline execution.
    """
    def __init__(self, nodes: Dict[str, Any], node_collection: Dict[str, Any]):
        self._node_collection = node_collection
        self._nodes = nodes

    def show(self, tablefmt=None):
        for k, v in self._node_collection.items():
            NodeVisualizer(k, v).show(tablefmt=tablefmt)
            print()

    @property
    def nodes(self):
        return list(self._node_collection.keys())

    def __getitem__(self, name: str):
        candidates = []
        for node in self.nodes:
            if re.search(name, node):
                candidates.append(node)

        if not candidates:
            engine_log.error('Node %s does not match any existing nodes. This pipeline contains following nodes: %s', name, self.nodes)
            raise KeyError('Node %s does not match any existing nodes. This pipeline contains following nodes: %s' % (name, self.nodes))
        if len(candidates) == 1:
            return NodeVisualizer(candidates[0], self._node_collection[candidates[0]])
        else:
            return [NodeVisualizer(i, self._node_collection[i]) for i in candidates]


class DataVisualizer:
    """
    DataVisualizer contains the data for a pipeline from one or several execution(s).
    """
    def __init__(self, nodes: Dict[str, Any], node_collection_list: List[Dict[str, Any]]):
        self._nodes = nodes
        self._node_collection_list = node_collection_list
        self._visualizers = [PipeVisualizer(nodes, i) for i in node_collection_list]

    def show(self, limit=1, tablefmt=None):
        limit = limit if limit > 0 else len(self._node_collection_list)
        for v in self._visualizers[:limit]:
            v.show(tablefmt=tablefmt)

    def __getitem__(self, idx):
        return self._visualizers[idx]

    def __len__(self):
        return len(self._visualizers)

    @property
    def visualizers(self):
        return self._visualizers

    @property
    def nodes(self):
        return list(self._visualizers[0].nodes)
