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
from typing import Dict, Any

from towhee.utils.log import engine_log
from towhee.datacollection import DataCollection


class NodeVisualizer:
    """
    Visualize the data of a node.
    """
    def __init__(self, name, data: Dict[str, Any]):
        self._name = name
        self._in = data['in']
        self._out = data['out']
        self._previous = data.get('previous', None)
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
    def op_input(self):
        return self._op_input

    def _show_base(self):
        print('Node:', self._name)
        print('Previous Node:', self._previous)
        print('Op_Input:', self._op_input)

    def _show_inputs(self, tablefmt):
        if isinstance(self._in, list):
            for idx, i in enumerate(self._in):
                print('Input from', self._previous[idx] + ':')
                i.show(tablefmt=tablefmt)
        else:
            if self._previous:
                print('Input from', self._previous[0] + ':')
            self._in.show(tablefmt=tablefmt)

    def _show_outputs(self, tablefmt):
        print('Output:')
        self._out.show(tablefmt=tablefmt)

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
    Visualize the data of the pipeline.
    """
    def __init__(self, dag_repr: 'DagRepr', node_queues: Dict[str, Any]):
        self._node_collections = node_queues
        self._dag_repr = dag_repr
        for node in self._dag_repr.nodes.values():
            if node.next_nodes:
                self._get_previous(node)
        for v in self._node_collections.values():
            if isinstance(v['in'], list):
                v['in'] = [DataCollection(i) for i in v['in']]
            else:
                v['in'] = DataCollection(v['in'])
            v['out'] = DataCollection(v['out'])

    def _get_previous(self, node):
        for i in node.next_nodes:
            curr = self._dag_repr.nodes[i].name
            if 'previous' not in self._node_collections[curr].keys():
                self._node_collections[curr]['previous'] = [node.name]
            else:
                self._node_collections[curr]['previous'].append(node.name)

    def show(self, tablefmt=None):
        for k in self._node_collections.keys():
            NodeVisualizer(k, self._node_collections[k]).show(tablefmt=tablefmt)
            print()

    @property
    def nodes(self):
        return list(self._node_collections.keys())

    def __getitem__(self, name: str):
        if name not in self._node_collections.keys():
            engine_log.error('Node %s does not exist. This pipeline contains following nodes: %s', name, self.nodes)
            raise KeyError('Node %s does not exist. This pipeline contains following nodes: %s' % (name, self.nodes))
        return NodeVisualizer(name, self._node_collections[name])
