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
from tabulate import tabulate

class GraphVisualizer:
    """
    Visualize a runtime pipeline with its strcuture and information.
    """
    def __init__(self, dag_repr: 'DAGRepr'):
        self._dag = dag_repr

    def _get_data(self):
        """
        Get dag info from DAG.
        """
        nodes = self._dag.nodes
        edges = self._dag.edges
        dag_dict = self._dag.dag_dict
        return [[
            nodes[k].name + '(' + dag_dict[k]['iter_info']['type'] + ')',
            [i[0] + ' (' + i[1].name[0] + ')' for i in edges[nodes[k].in_edges[0]]['data'] if i[0] in nodes[k].inputs],
            [i[0] + ' (' + i[1].name[0] + ')' for i in edges[nodes[k].out_edges[0]]['data'] if i[0] in nodes[k].outputs],
            [i[0] + ' (' + i[1].name[0] + ')' for i in edges[nodes[k].in_edges[0]]['data']],
            [i[0] + ' (' + i[1].name[0] + ')' for i in edges[nodes[k].out_edges[0]]['data']],
            None if not dag_dict[k]['next_nodes'] else [nodes[i].name for i in dag_dict[k]['next_nodes']],
            dag_dict[k]['iter_info']['param']
        ] for k in self._dag.get_top_sort(nodes)]

    def show(self):
        headers = [
            'node',
            'op_inputs',
            'op_outputs',
            'node_inputs',
            'node_outputs',
            'next',
            'iter_param'
        ]

        print(tabulate(self._get_data(), headers=headers))
