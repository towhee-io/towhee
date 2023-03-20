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
import uuid
import json
from copy import deepcopy
from typing import Dict, Any, Set, List, Tuple

from towhee.runtime.check_utils import check_set, check_node_iter
from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.schema_repr import SchemaRepr
from towhee.runtime.node_config import TowheeConfig
from towhee.runtime.constants import (
    WindowAllConst,
    WindowConst,
    FilterConst,
    TimeWindowConst,
    FlatMapConst,
    InputConst,
    OutputConst,
    OPType,
)


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
        dag_dict(`Dict`): The dag dict.
    """
    def __init__(self, nodes: Dict[str, NodeRepr], edges: Dict[int, Dict], dag_dict: Dict[str, Any] = None, top_sort: list = None):
        self._nodes = nodes
        self._edges = edges
        self._dag_dict = dag_dict
        if not top_sort:
            self._top_sort = self.get_top_sort(nodes)
        else:
            self._top_sort = top_sort

    @property
    def nodes(self) -> Dict:
        return self._nodes

    @property
    def edges(self) -> Dict:
        return self._edges

    @property
    def dag_dict(self) -> Dict:
        return self._dag_dict

    @property
    def top_sort(self) -> list:
        return self._top_sort

    @staticmethod
    def check_nodes(nodes: Dict[str, NodeRepr], top_sort: list):
        """Check nodes if start with _input and ends with _output, and the schema has declared before using.

        Args:
            nodes (`Dict[str, NodeRepr]`): All the nodes from DAG.
            top_sort (`list`): Topological list.
        """
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
        outputs_schema = ()
        while stack:
            n = stack.pop()
            check_schema = nodes[n].inputs
            used_col = DAGRepr.get_base_col(nodes[n])
            if used_col is not None:
                if isinstance(used_col, str):
                    check_schema += (used_col,)
                else:
                    check_schema += tuple(used_col)

            common_schema = (set(check_schema)-set(outputs_schema)) & ahead_schema
            for x in common_schema:
                ahead_schema.remove(x)
                used_schema.add(x)
            next_nodes = nodes[n].next_nodes
            if len(ahead_schema) == 0:
                break
            if next_nodes is None:
                continue

            for i in next_nodes[::-1]:
                if i not in visited:
                    stack.append(i)
            visited.append(n)
            outputs_schema += nodes[n].outputs

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
                if iter_type == 'concat':
                    inputs_type = [ahead_schemas[d].type]
                else:
                    inputs_type = [ahead_schemas[inp].type for inp in inputs]
                    inputs_type.append(ahead_schemas[d].type)
                edge_schemas[d] = SchemaRepr.from_dag(d, iter_type, inputs_type)
            else:
                edge_schemas[d] = SchemaRepr.from_dag(d, 'map', [ahead_schemas[d].type])
        edge = {'schema': edge_schemas, 'data': [(s, t.type) for s, t in edge_schemas.items()]}
        return edge

    @staticmethod
    def set_edges(nodes: Dict[str, NodeRepr], top_sort: list):
        """Set in_edges and out_edges for the node, and return the nodes and edge.

        Args:
            nodes (`Dict[str, NodeRepr]`): All the nodes repr from DAG.
            top_sort (`list`): Topological list.

        Returns:
            Dict[str, NodeRepr]: The nodes update in_edges and out_edges.
            Dict[str, Dict]: The edges for the DAG.
        """
        out_id = 0
        edges = {out_id: DAGRepr.get_edge_from_schema(nodes[InputConst.name].outputs, nodes[InputConst.name].inputs, nodes[InputConst.name].outputs,
                                                      nodes[InputConst.name].iter_info.type, None)}
        nodes[InputConst.name].in_edges = [out_id]

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

    def to_dict(self):
        info = {}
        info['edges'] = {}
        info['nodes'] = {}

        for k, v in self._edges.items():
            info['edges'][k] = []
            for (name, ctype) in v['data']:
                info['edges'][k].append({'name': name, 'type': ctype.name})

        for k, v in self._nodes.items():
            info['nodes'][k] = {}
            info['nodes'][k]['name'] = v.name
            info['nodes'][k]['iter_info'] = {'type': v.iter_info.type, 'param': v.iter_info.param}
            info['nodes'][k]['op_info'] = {
                'operator': v.op_info.operator.__name__ if callable(v.op_info.operator) else v.op_info.operator,
                'type': v.op_info.type
            }
            info['nodes'][k]['next_nodes'] = v.next_nodes
            info['nodes'][k]['inputs'] = v.in_edges
            info['nodes'][k]['outputs'] = v.out_edges
            info['nodes'][k]['op_input'] = v.inputs
            info['nodes'][k]['op_output'] = v.outputs

        return info

    def to_json(self, **kws):
        return json.dumps(self.to_dict(), **kws)

    @staticmethod
    def from_dict(dag: Dict[str, Any]):
        """Return a DAGRepr from a dag dictionary.

        Args:
            dag (`str`): The dag dictionary.

        Returns:
            DAGRepr
        """
        dag_dict = deepcopy(dag)
        for uid, node in dag_dict.items():
            if node['op_info']['type'] == OPType.PIPELINE:
                DAGRepr.rebuild_dag(dag, uid, node['op_info'], node['iter_info'], node['inputs'], node['outputs'])

        def _get_name(val):
            if val['op_info']['type'] == OPType.CALLABLE:
                if isinstance(val['op_info']['operator'], types.FunctionType):
                    name = val['op_info']['operator'].__name__
                else:
                    name = type(val['op_info']['operator']).__name__
            elif val['op_info']['type'] == OPType.LAMBDA:
                name = 'lambda'
            elif val['op_info']['type'] in [OPType.HUB, OPType.BUILTIN]:
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
            if 'config' in val and val['config'] is not None and isinstance(val['config'], TowheeConfig):
                val['config'] = val['config'].config
                assert isinstance(val['config'], dict)

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
                elif val['config']['name'] in [InputConst.name, OutputConst.name]:
                    val['config']['name'] = val['config']['name'] + '-' + str(node_index)
                node_index += 1

            nodes[key] = NodeRepr.from_dict(key, val)
        top_sort = DAGRepr.get_top_sort(nodes)
        DAGRepr.check_nodes(nodes, top_sort)
        dag_nodes, schema_edges = DAGRepr.set_edges(nodes, top_sort)
        return DAGRepr(dag_nodes, schema_edges, dag, top_sort)

    @staticmethod
    def rebuild_dag(dag, sub_uid, op_info, iter_info, input_schema, output_schema):
        sub_dag = op_info['dag']
        sub_top_sort = op_info['top_sort']
        iter_type = iter_info['type']
        param = iter_info['param']

        used_schemas = ()
        for _, node in dag.items():
            used_schemas += node['inputs'] + node['outputs']
        if iter_type == FilterConst.name:
            filter_cols = param[FilterConst.param.filter_by]
            new_dag, sub_uid_out = DAGRepr.rebuild_sub_dag(sub_dag, sub_uid, sub_top_sort, filter_cols, filter_cols, used_schemas)
        else:
            new_dag, sub_uid_out = DAGRepr.rebuild_sub_dag(sub_dag, sub_uid, sub_top_sort, input_schema, output_schema, used_schemas)
        new_dag[sub_uid_out]['next_nodes'] += dag[sub_uid]['next_nodes']

        if iter_type == FlatMapConst.name:
            new_dag[sub_uid_out]['iter_info'] = {'type': FlatMapConst.name,
                                                 'param': None}
        elif iter_type == FilterConst.name:
            new_dag[sub_uid_out]['iter_info'] = {'type': FilterConst.name,
                                                 'param': {FilterConst.param.filter_by: new_dag[sub_uid_out]['inputs']}}
            new_dag[sub_uid_out]['inputs'] = input_schema
            new_dag[sub_uid_out]['outputs'] = output_schema
        elif iter_type == WindowConst.name:
            new_dag[sub_uid]['iter_info'] = {'type': WindowConst.name,
                                             'param': {WindowConst.param.size: param[WindowConst.param.size],
                                                       WindowConst.param.step: param[WindowConst.param.step]}}
        elif iter_type == TimeWindowConst.name:
            new_dag[sub_uid]['iter_info'] = {'type': TimeWindowConst.name,
                                             'param': {TimeWindowConst.param.time_range_sec: param[TimeWindowConst.param.time_range_sec],
                                                       TimeWindowConst.param.time_step_sec: param[TimeWindowConst.param.time_step_sec],
                                                       TimeWindowConst.param.timestamp_col: param[TimeWindowConst.param.timestamp_col]}}
        elif iter_type == WindowAllConst.name:
            new_dag[sub_uid]['iter_info'] = {'type': WindowAllConst.name,
                                             'param': None}

        for _, node in new_dag.items():
            if node['config']['name'].startswith(InputConst.name) or node['config']['name'].startswith(OutputConst.name):
                node['config']['name'] = node['config']['name'].split('-')[0]
            else:
                node['config'].pop('name')
        dag.update(new_dag)

    @staticmethod
    def rebuild_sub_dag(sub_dag, uid_in, sub_top_sort, input_schema, output_schema, used_schemas):
        pipe_dag = deepcopy(sub_dag)

        # update the duplicate schemas in pipe_dag
        DAGRepr._rename_schemas(pipe_dag, set(used_schemas))
        if len(input_schema) != len(pipe_dag['_input']['inputs']):
            DAGRepr._rename_group_schemas(pipe_dag, sub_top_sort, pipe_dag['_input']['inputs'], input_schema)
        # update input node to pipe_dag
        DAGRepr._update_input(pipe_dag, input_schema, uid_in)
        # update output node to pipe_dag
        uid_out = uuid.uuid4().hex
        DAGRepr._update_output(pipe_dag, output_schema, uid_out, sub_top_sort[-2])

        return pipe_dag, uid_out

    @staticmethod
    def _update_input(dag, input_schema, uid):
        input_info = dag.pop('_input')
        input_info['outputs'] = input_info['inputs']
        input_info['inputs'] = input_schema
        dag[uid] = input_info

    @staticmethod
    def _update_output(dag, output_schema, uid, mark_node):
        output_info = dag.pop('_output')
        output_info['inputs'] = output_info['outputs']
        output_info['outputs'] = output_schema
        dag[uid] = output_info
        dag[mark_node]['next_nodes'].remove('_output')
        dag[mark_node]['next_nodes'].append(uid)

    @staticmethod
    def _rename_group_schemas(dag, top_sort, ori_input_schema, input_schema):
        assert len(input_schema) == 1 or len(ori_input_schema) == 1
        for node_uid in top_sort:
            dag[node_uid]['inputs'] = DAGRepr._replace_schema(dag[node_uid]['inputs'], ori_input_schema, input_schema)
            DAGRepr._replace_cols(dag[node_uid]['iter_info']['type'], dag[node_uid]['iter_info']['param'], ori_input_schema, input_schema)

            ori_name = DAGRepr._to_string(ori_input_schema)
            new_name = DAGRepr._to_string(dag[node_uid]['outputs'])
            if ori_name in new_name and node_uid != '_input':
                break

    @staticmethod
    def _rename_schemas(dag, schemas):
        for schema in schemas:
            for _, node in dag.items():
                node['inputs'] = DAGRepr._replace_schema(node['inputs'], schema)
                node['outputs'] = DAGRepr._replace_schema(node['outputs'], schema)
                DAGRepr._replace_cols(node['iter_info']['type'], node['iter_info']['param'], schema)

    @staticmethod
    def _replace_cols(node_type, node_param, schema, new_schema=None):
        if node_type == FilterConst.name:
            node_param[FilterConst.param.filter_by] = DAGRepr._replace_schema(node_param[FilterConst.param.filter_by], schema, new_schema)
        elif node_type == TimeWindowConst.name:
            node_param[TimeWindowConst.param.timestamp_col] = DAGRepr._replace_schema((node_param[TimeWindowConst.param.timestamp_col],),
                                                                                      schema, new_schema)[0]

    @staticmethod
    def _replace_schema(schema, ori_name, new_name=None):
        if not new_name:
            new_name = ori_name + '_bak'
            new_schema = [new_name if name == ori_name else name for name in schema]
        else:
            ori_name = DAGRepr._to_string(ori_name)
            new_name = DAGRepr._to_string(new_name)
            new_schema = DAGRepr._to_string(schema).replace(ori_name, new_name)
            new_schema = new_schema.split(',')[1:-1]
        return tuple(new_schema)

    @staticmethod
    def _to_string(schema):
        str_schema = ','
        for x in schema:
            str_schema += x + ','
        return str_schema
