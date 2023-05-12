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
        self._data = data
        self._name = name
        self._previous = data.get('previous')
        self._next = data.get('next')
        self._in = data['in']
        self._out = data['out']
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

    def _prepare_data(self, show_inputs: bool, show_outputs: bool):
        headers = ['NodeInfo']
        data = [{
                'headers': ['item', 'info'],
                'data': [
                    ['NodeName', self._name],
                    ['NodeType', self._data.get('type')],
                    ['InputSchema', self._data.get('op_input')],
                    ['OutputSchema', self._data.get('op_output')],
                    ['Operator', self._data.get('operator')]
                ]
            }]

        if show_inputs:
            headers.append('Inputs')
            data.append({
                'headers': self._previous if self._previous else ['PipeInput'],
                'data': [[dc.prepare_table_data(-1) for dc in self._in]],
            })
        if show_outputs:
            headers.append('Outputs')
            data.append({
                'headers': self._next if self._next else ['PipeOutput'],
                'data': [[dc.prepare_table_data(-1) for dc in self._out]],
            })
        return {'headers': headers, 'data': [data]}

    def show_inputs(self):
        try:
            # check is ipython
            get_ipython()
            from towhee.utils.html_table import NestedHTMLTable # pylint: disable=import-outside-toplevel
            NestedHTMLTable(self._prepare_data(True, False)).show()
        except NameError:
            from towhee.utils.console_table import NestedConsoleTable # pylint: disable=import-outside-toplevel
            NestedConsoleTable(self._prepare_data(True, False)).show()

    def show_outputs(self):
        try:
            # check is ipython
            get_ipython()
            from towhee.utils.html_table import NestedHTMLTable # pylint: disable=import-outside-toplevel
            NestedHTMLTable(self._prepare_data(False, True)).show()
        except NameError:
            from towhee.utils.console_table import NestedConsoleTable # pylint: disable=import-outside-toplevel
            NestedConsoleTable(self._prepare_data(False, True)).show()

    def show(self):
        try:
            # check is ipython
            get_ipython()
            from towhee.utils.html_table import NestedHTMLTable # pylint: disable=import-outside-toplevel
            NestedHTMLTable(self._prepare_data(True, True)).show()
        except NameError:
            from towhee.utils.console_table import NestedConsoleTable # pylint: disable=import-outside-toplevel
            NestedConsoleTable(self._prepare_data(True, True)).show()


class PipeVisualizer:
    """
    Visualize the data from single pipeline execution.
    """
    def __init__(self, nodes: Dict[str, Any], node_collection: Dict[str, Any]):
        self._node_collection = node_collection
        self._nodes = nodes

    def show(self):
        for k, v in self._node_collection.items():
            NodeVisualizer(k, v).show()

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

    def show(self, limit=1):
        limit = limit if limit > 0 else len(self._node_collection_list)
        for v in self._visualizers[:limit]:
            v.show()

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
