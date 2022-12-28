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

import types
from typing import Dict, Any, Set, List, Tuple

from towhee.runtime.check_utils import check_set, check_node_iter
from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.schema_repr import SchemaRepr
from towhee.runtime.constants import FilterConst, TimeWindowConst, OPType, InputConst, OutputConst


class DAGRepr:
    """
    A `DAGRepr` represents a complete DAG.

    Args:
        nodes (`Dict[str, NodeRepr]`): All nodes in the dag, which start with _input and end with _output node.
        edges (`Dict[str, List]`): The edges about data queue schema, such as:
                            { 0: {'data': [(a, ColumnType.SCALAR), (b, ColumnType.SCALAR)], 'schema': {'a', SchemaRepr, 'b', SchemaRepr}}
                              1: {'data': [(b, ColumnType.SCALAR), (c, ColumnType.SCALAR)], 'schema': {'b', SchemaRepr, 'c', SchemaRepr}}
                              2: {'data': [(a, ColumnType.SCALAR), (c, ColumnType.SCALAR)], 'schema': {'a', SchemaRepr, 'c', SchemaRepr}}
                            }
    """
    def __init__(self, nodes: Dict[str, NodeRepr], edges: Dict[int, Dict]):
        self._nodes = nodes
        self._edges = edges

    @property
    def nodes(self) -> Dict:
        return self._nodes

    @property
    def edges(self) -> Dict:
        return self._edges

    @staticmethod
    def check_nodes(nodes: Dict[str, NodeRepr]):
        """Check nodes if start with _input and ends with _output, and the schema has declared before using.

        Args:
            nodes (`Dict[str, NodeRepr]`): All the nodes from DAG.
        """
        top_sort = DAGRepr.get_top_sort(nodes)
        if len(top_sort) != len(nodes):
            raise ValueError('The DAG is not valid, it has a circle.')
        if top_sort[0] != InputConst.name:
            raise ValueError('The DAG is not valid, it does not started with `_input`.')
        if top_sort[-1] != OutputConst.name:
            raise ValueError('The DAG is not valid, it does not ended with `_output`.')

        all_inputs = DAGRepr.get_all_inputs(nodes, top_sort)
        for name in top_sort[1:]:
            check_set(nodes[name].inputs, set(all_inputs[name]))
            check_node_iter(nodes[name].iter_info.type, nodes[name].iter_info.param,
                            nodes[name].inputs, nodes[name].outputs, set(all_inputs[name]))

    @staticmethod
    def get_top_sort(nodes: Dict[str, NodeRepr]):
        """Get the topological order of the DAG.

        Args:
            nodes (`Dict[str, NodeRepr]`): All the nodes repr from DAG.

        Returns:
           List: Topological list.
        """
        graph = dict((name, nodes[name].next_nodes) for name in nodes)
        if OutputConst.name in graph:
            graph[OutputConst.name] = []
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
    def get_all_inputs(nodes: Dict[str, NodeRepr], top_sort: list):
        """Get all the inputs of the dag nodes, include ahead and current nodes.

        Args:
            nodes (`Dict[str, NodeRepr]`): All the nodes repr from DAG.
            top_sort (`list`): Topological list.

        Returns:
           Dict[str, Tuple]: Dict of the node and the all inputs of this node.
        """
        all_nodes = [InputConst.name]
        all_inputs = dict((name, nodes[InputConst.name].outputs) for name in nodes)
        for name in top_sort[1:]:
            if nodes[name].iter_info.type == 'concat':
                pre_name = all_nodes.pop()
                while name in nodes[pre_name].next_nodes:
                    nodes[name].inputs += nodes[pre_name].outputs
                    if len(all_nodes) == 0:
                        break
                    pre_name = all_nodes.pop()
                nodes[name].outputs = nodes[name].inputs
            all_nodes.append(name)
            if nodes[name].next_nodes is None:
                continue
            for n in nodes[name].next_nodes:
                all_inputs[n] = all_inputs[n] + nodes[name].outputs + all_inputs[name]
        return all_inputs

    @staticmethod
    def dfs_used_schema(nodes: Dict[str, NodeRepr], name: str, ahead_edge: Set):
        """Get the used schema behind the node.

        Args:
            nodes (`Dict[str, NodeRepr]`): All the nodes from DAG.
            name (`str`): Current node name.
            ahead_edge (`Set`): As set of the ahead edge schema.

        Returns:
           Set: A set of the used schema behind the node.
        """
        ahead_schema = ahead_edge.copy()
        used_schema = set()
        stack = [name]
        visited = [name]
        while stack:
            n = stack.pop()
            used_col = DAGRepr.get_base_col(nodes[n])
            if used_col is not None:
                if isinstance(used_col, str):
                    used_schema.add(used_col)
                else:
                    for c in used_col:
                        used_schema.add(c)

            common_schema = set(nodes[n].inputs) & ahead_schema
            for x in common_schema:
                ahead_schema.remove(x)
                used_schema.add(x)
            next_nodes = nodes[n].next_nodes
            if len(ahead_schema) == 0 or next_nodes is None:
                break

            for i in next_nodes[::-1]:
                if i not in visited:
                    stack.append(i)
                    visited.append(i)
        return used_schema

    @staticmethod
    def get_base_col(nodes: NodeRepr):
        if nodes.iter_info.type == FilterConst.name:
            return nodes.iter_info.param[FilterConst.param.filter_by]
        if nodes.iter_info.type == TimeWindowConst.name:
            return nodes.iter_info.param[TimeWindowConst.param.timestamp_col]
        return None

    @staticmethod
    def get_edge_from_schema(schema: Tuple, inputs: Tuple, outputs: Tuple, iter_type: str, ahead_edges: List) -> Dict:
        """Return the edge form the schema info for the node.

        Args:
            schema (`Tuple): The out edge schema of this node.
            inputs (`Tuple`): The inputs of this node.
            outputs (`Tuple`): The outputs of this node.
            iter_type (`str`): The iteration type of this node.
            ahead_edges (`list`): A list of the ahead edges.

        Returns:
           Dict[str, Dict]: A edge include data and schema.
        """
        if inputs is None:
            inputs = outputs
        if ahead_edges is None:
            edge_schemas = dict((d, SchemaRepr.from_dag(d, iter_type)) for d in schema)
            edge = {'schema': edge_schemas, 'data': [(s, t.type) for s, t in edge_schemas.items()]}
            return edge
        ahead_schemas = {}
        for ahead in ahead_edges:
            ahead_schemas.update(ahead)

        edge_schemas = {}
        for d in schema:
            if d not in ahead_schemas:
                inputs_type = [ahead_schemas[inp].type for inp in inputs]
                edge_schemas[d] = SchemaRepr.from_dag(d, iter_type, inputs_type)
            elif d in outputs:
                edge_schemas[d] = SchemaRepr.from_dag(d, iter_type, [ahead_schemas[d].type])
            else:
                edge_schemas[d] = SchemaRepr.from_dag(d, 'map', [ahead_schemas[d].type])
        edge = {'schema': edge_schemas, 'data': [(s, t.type) for s, t in edge_schemas.items()]}
        return edge

    @staticmethod
    def set_edges(nodes: Dict[str, NodeRepr]):
        """Set in_edges and out_edges for the node, and return the nodes and edge.

        Args:
            nodes (`Dict[str, NodeRepr]`): All the nodes repr from DAG.

        Returns:
            Dict[str, NodeRepr]: The nodes update in_edges and out_edges.
            Dict[str, Dict]: The edges for the DAG.
        """
        out_id = 0
        edges = {out_id: DAGRepr.get_edge_from_schema(nodes[InputConst.name].outputs, nodes[InputConst.name].inputs, nodes[InputConst.name].outputs,
                                                      nodes[InputConst.name].iter_info.type, None)}
        nodes[InputConst.name].in_edges = [out_id]

        top_sort = DAGRepr.get_top_sort(nodes)
        for name in top_sort[:-1]:
            ahead_schema = set(nodes[name].outputs)
            for i in nodes[name].in_edges:
                ahead_schema = ahead_schema | edges[i]['schema'].keys()
            for next_name in nodes[name].next_nodes:
                out_id += 1
                out_schema = DAGRepr.dfs_used_schema(nodes, next_name, ahead_schema)
                edges[out_id] = DAGRepr.get_edge_from_schema(tuple(out_schema), nodes[name].inputs, nodes[name].outputs,
                                                             nodes[name].iter_info.type, [edges[e]['schema'] for e in nodes[name].in_edges])

                if nodes[next_name].in_edges is None:
                    nodes[next_name].in_edges = [out_id]
                else:
                    nodes[next_name].in_edges.append(out_id)

                if nodes[name].out_edges is None:
                    nodes[name].out_edges = [out_id]
                else:
                    nodes[name].out_edges.append(out_id)

        out_id += 1
        final_edge = nodes[OutputConst.name].in_edges[0]
        final_schema = edges[final_edge]['schema']
        edges[out_id] = {'schema': final_schema, 'data': [(s, final_schema[s].type) for s in nodes[OutputConst.name].inputs]}

        nodes[OutputConst.name].out_edges = [out_id]
        return nodes, edges

    @staticmethod
    def from_dict(dag: Dict[str, Any]):
        """Return a DAGRepr from a dag dictionary.

        Args:
            dag (`str`): The dag dictionary.

        Returns:
            DAGRepr
        """
        def _get_name(val):
            if val['op_info']['type'] == OPType.CALLABLE:
                if isinstance(val['op_info']['operator'], types.FunctionType):
                    name = val['op_info']['operator'].__name__
                else:
                    name = type(val['op_info']['operator']).__name__
            elif val['op_info']['type'] == OPType.LAMBDA:
                name = 'lambda'
            elif val['op_info']['type'] == OPType.HUB:
                fn = val['op_info']['operator']
                if isinstance(fn, str):
                    name = fn
                else:
                    name = fn.__class__.__name__

            return name

        nodes = {}
        node_index = 0
        for key, val in dag.items():
            # Deal with AutoConfig
            if 'config' in val and val['config'] is not None and isinstance(val['config'], list):
                dict_config = {}
                for conf in val['config']:
                    assert isinstance(conf, dict)
                    dict_config.update(conf)
                val['config'] = dict_config

            # Deal with input and output.
            if key in [InputConst.name, OutputConst.name]:
                val['config'] = {'name': key}

            # Concat nodes does not have op_info.
            elif val['iter_info']['type'] == 'concat':
                val['config'] = {'name': 'concat-' + str(node_index)}
                node_index += 1

            # If config does not specified.
            elif 'config' not in val or not val['config']:
                name = _get_name(val)
                val['config'] = {'name': name + '-' + str(node_index)}
                node_index += 1

            # Process dict config.
            elif isinstance(val['config'], dict):
                if 'name' not in val['config']:
                    name = _get_name(val)
                    val['config']['name'] = name + '-' + str(node_index)
                    node_index += 1

            nodes[key] = NodeRepr.from_dict(key, val)
        DAGRepr.check_nodes(nodes)
        dag_nodes, schema_edges = DAGRepr.set_edges(nodes)
        return DAGRepr(dag_nodes, schema_edges)
