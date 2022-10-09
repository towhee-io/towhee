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

from typing import Dict, List
from towhee.runtime.node_repr import NodeRepr


class DAGRepr:
    """
    A `DAGRepr` represents a complete DAG.

    Args:
        dag (`Dict[str, Any]`): The DAG dictionary.
    """
    def __init__(self, dag: dict):
        self._dag = dag

    @property
    def dag(self) -> Dict:
        return self._dag

    def get_nodes(self) -> List:
        """
        Get a list of NodeRepr from the DAG.

        Examples:
        >>> from towhee.runtime.dag_repr import DAGRepr
        >>> towhee_dag = {
        ...  '_input': {'inputs': ('a', 'b'), 'outputs': ('a', 'b'), 'fn_type': '_input', 'iteration': 'map'},
        ...  'e433a': {'function': 'towhee.decode', 'init_args': ('a',), 'init_kws': {'b': 'b'}, 'inputs': 'a', 'outputs': 'c', 'fn_type': 'hub',
        ...            'iteration': 'map', 'config': None, 'tag': 'main', 'param': None},
        ...  'b1196': {'function': 'towhee.hub', 'init_args': ('a',), 'init_kws': {'b': 'b'}, 'inputs': ('a', 'b'), 'outputs': 'b', 'fn_type': 'hub',
        ...            'iteration': 'filter', 'config': None, 'tag': '1.1', 'param': {'filter_columns': 'a'}},
        ...  '_output': {'inputs': 'd', 'outputs': 'd', 'fn_type': '_output', 'iteration': 'map'},
        ... }
        >>> dr = DAGRepr(towhee_dag)
        >>> nodes = dr.get_nodes()
        >>> print(nodes[0].name, nodes[2].function, nodes[2].iteration, nodes[2].param, nodes[3].name)
        _input towhee.clip filter {'filter_columns': 'a'} _output
        """
        nodes_list = []
        for name in self._dag:
            nodes_list.append(NodeRepr.from_dict(name, self._dag[name]))
        return nodes_list
