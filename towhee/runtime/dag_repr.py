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

from typing import Dict, Any, Set

from towhee.runtime.node_repr import NodeRepr


def check_set(base: Set[str], parent: Set[str], equal: bool = False):
    """
    Check if the src is a valid input and output.

    Args:
        base (`Dict[str, Any]`): The base set will be check.
        parent (`Set[str]`): The parents set to check.
        equal (`bool`): Whether to check if two sets are equal

    Returns:
        (`bool | raise`)
            Return `True` if it is valid, else raise exception.
    """
    if equal and base != parent:
        raise ValueError(f'The DAG Nodes inputs {str(base)} is not equal to the output: {parent}')
    elif not base.issubset(parent):
        raise ValueError(f'The DAG Nodes inputs {str(base)} is not valid, this is not declared: {base - parent}.')


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
            raise ValueError(f'The DAG Nodes is not valid, it does not started with `_input`.')
        check_set(set(nodes.inputs), set(nodes.outputs), equal=True)

        all_inputs = set(nodes.inputs)
        nodes = nodes.next_node
        while nodes.next_node is not None:
            inputs = set(nodes.inputs)
            check_set(inputs, all_inputs)
            for i in nodes.outputs:
                all_inputs.add(i)
            nodes = nodes.next_node

        if nodes.name != '_output':
            raise ValueError(f'The DAG Nodes is not valid, it does not ended with `_output`.')
        check_set(set(nodes.inputs), all_inputs)
        check_set(set(nodes.inputs), set(nodes.outputs), equal=True)

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
