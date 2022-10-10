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

from towhee.runtime.node_repr import NodeRepr


class DAGRepr:
    """
    A `DAGRepr` represents a complete DAG.

    Args:
        nodes (`NodeRepr`): All nodes in the dag, which start with _input and end with _output node, and nodes.next_node will show more process.
        dag_type (`str`): The type of DAG type, such as 'local', 'triton' and 'mix', defaults to 'local'.
    """
    def __init__(self, nodes: NodeRepr, dag_type: str = 'local'):
        self._nodes = nodes
        self._dag_type = dag_type

    @property
    def nodes(self) -> NodeRepr:
        return self._nodes

    @property
    def dag_type(self) -> str:
        return self._dag_type

    @staticmethod
    def check_nodes(nodes: NodeRepr):
        if nodes.name != '_input':
            raise ValueError(f'The DAG Nodes {str(nodes)} is not valid, it does not started with `_input`.')
        all_inputs = set(nodes.inputs)
        while nodes.next_node is not None:
            inputs = set(nodes.inputs)
            if not inputs.issubset(all_inputs):
                raise ValueError(f'The DAG Nodes {str(nodes)} is not valid, the outputs is not declared: {inputs - all_inputs}.')
            for i in nodes.outputs:
                all_inputs.add(i)
            nodes = nodes.next_node
        if nodes.name != '_output':
            raise ValueError(f'The DAG Nodes {str(nodes)} is not valid, it does not ended with `_output`.')
        else:
            if nodes.inputs != nodes.outputs:
                raise ValueError(f'The DAG _output Nodes {str(nodes)} is not valid, the input: {nodes.inputs} is not equal output: {nodes.outputs}.')

    @staticmethod
    def from_dict(dag: Dict[str, Any], dag_type: str = 'local'):
        reversed_dag_keys = list(reversed(dag.keys()))
        dag_nodes = NodeRepr.from_dict(reversed_dag_keys[0], dag[reversed_dag_keys[0]])
        for node in reversed_dag_keys[1:]:
            one_node = NodeRepr.from_dict(node, dag[node])
            one_node.next_node = dag_nodes
            dag_nodes = one_node
        DAGRepr.check_nodes(dag_nodes)
        return DAGRepr(dag_nodes, dag_type)
