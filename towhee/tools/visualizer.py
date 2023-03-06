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

class Visualizer:
    """
    Visualize a runtime pipeline with its strcuture and information.
    """
    def __init__(self, dag_repr: 'DAGRepr'):
        self._dag = dag_repr

    def _get_data(self):
        """
        Get dag info from DAG.
        """
        return [[
            self._dag.nodes[k].name + '(' + v['iter_info']['type'] + ')',
            [i[0] + '(' + i[1].name + ')' for i in self._dag.edges[self._dag.nodes[k].in_edges[0]]['data'] if i[0] in self._dag.nodes[k].inputs],
            [i[0] + '(' + i[1].name + ')' for i in self._dag.edges[self._dag.nodes[k].out_edges[0]]['data'] if i[0] in self._dag.nodes[k].outputs],
            [i[0] + '(' + i[1].name + ')' for i in self._dag.edges[self._dag.nodes[k].in_edges[0]]['data']],
            [i[0] + '(' + i[1].name + ')' for i in self._dag.edges[self._dag.nodes[k].out_edges[0]]['data']],
            None if not v['next_nodes'] else [self._dag.nodes[i].name for i in v['next_nodes']],
            v['iter_info']['param']
        ] for k, v in self._dag.dag_dict.items()]

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
