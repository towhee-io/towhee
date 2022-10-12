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
        raise ValueError(f'The DAG Nodes inputs {str(base)} is not valid, which is not declared: {base - parent}.')


class DAGRepr:
    """
    A `DAGRepr` represents a complete DAG.

    Args:
        nodes (`Dict[str, NodeRepr]`): All nodes in the dag, which start with _input and end with _output node.
        schema_edges (`Dict[str, Set]`): The edges about data queue schema.
    """
    def __init__(self, nodes: Dict[str, NodeRepr], schema_edges: Dict[str, Set]):
        self._nodes = nodes
        self._schema_edges = schema_edges

    @property
    def nodes(self) -> Dict:
        return self._nodes

    @property
    def schema_edges(self) -> Dict:
        return self._schema_edges

    @staticmethod
    def check_nodes(nodes: Dict[str, NodeRepr]):
        top_sort = DAGRepr.get_top_sort(nodes)
        if top_sort[0] != '_input':
            raise ValueError('The DAG is not valid, it does not started with `_input`.')
        if top_sort[-1] != '_output':
            raise ValueError('The DAG is not valid, it does not ended with `_output`.')
        if len(top_sort) != len(nodes):
            raise ValueError('The DAG is not valid, it has a circle.')

        all_inputs = DAGRepr.get_all_inputs(nodes)
        for name in nodes:
            check_set(set(nodes[name].inputs), set(all_inputs[name]))
        check_set(set(nodes['_input'].inputs), set(nodes['_input'].inputs), True)
        check_set(set(nodes['_output'].inputs), set(nodes['_output'].outputs), True)

    @staticmethod
    def get_top_sort(nodes: Dict[str, NodeRepr]):
        graph = dict((name, nodes[name].next_nodes) for name in nodes)
        result = []
        while True:
            temp_list = {j for i in graph.values() for j in i}
            node = [x for x in (list(graph.keys())) if x not in temp_list]
            if not node:
                break
            result.append(node[0])
            del graph[node[0]]
        return result

    @staticmethod
    def get_all_inputs(nodes: Dict[str, NodeRepr]):
        if '_input' not in nodes.keys():
            raise ValueError('The DAG Nodes is not valid, it does not have key `_input`.')
        all_inputs = dict((name, nodes['_input'].inputs) for name in nodes)
        for name in nodes:
            for n in nodes[name].next_nodes:
                all_inputs[n] += nodes[name].outputs
        return all_inputs

    @staticmethod
    def dfs_used_schema(nodes: Dict[str, NodeRepr], name: str, ahead_schema: Set):
        used_schema = set()
        stack = list(nodes[name].next_nodes)
        visited = stack[-1:]
        while stack:
            n = stack.pop()
            common_schema = set(nodes[n].inputs) & ahead_schema
            for x in common_schema:
                ahead_schema.remove(x)
                used_schema.add(x)

            next_nodes = nodes[n].next_nodes
            for i in next_nodes[::-1]:
                if i not in visited:
                    stack.append(i)
                    visited.append(i)
            if len(ahead_schema) == 0:
                break
        return used_schema

    @staticmethod
    def set_schema_edges(nodes: Dict[str, NodeRepr]):
        top_sort = DAGRepr.get_top_sort(nodes)
        edges = {}
        for name in top_sort:
            if name == '_input':
                edges['0'] = set(nodes[name].inputs)
                out_edge = set(nodes[name].outputs)
            elif name == '_output':
                out_edge = set(nodes[name].outputs)
            else:
                ahead_schema = set()
                for i in nodes[name].in_edges:
                    ahead_schema = ahead_schema | edges[i]
                used_schema = DAGRepr.dfs_used_schema(nodes, name, ahead_schema)
                out_edge = set(nodes[name].outputs) | used_schema

            out_id = None
            for e_id, schema in edges.items():
                if out_edge == schema:
                    out_id = e_id
                    break
            if out_id is None:
                out_id = str(len(edges))
                edges[out_id] = out_edge

            if name == '_input':
                nodes[name].in_edges = [out_id]
            nodes[name].out_edges = [out_id]
            for next_node in nodes[name].next_nodes:
                cur_node = nodes[next_node]
                cur_node.in_edges = cur_node.in_edges + [out_id]
        return nodes, edges

    @staticmethod
    def from_dict(dag: Dict[str, Any]):
        nodes = dict((name, NodeRepr.from_dict(name, dag[name])) for name in dag)
        DAGRepr.check_nodes(nodes)
        dag_nodes, schema_edges = DAGRepr.set_schema_edges(nodes)
        return DAGRepr(dag_nodes, schema_edges)
