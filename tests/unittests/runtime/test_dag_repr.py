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
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import unittest

from towhee import pipe
from towhee.runtime.dag_repr import DAGRepr, NodeRepr
from towhee.runtime.data_queue import ColumnType
from towhee.runtime.schema_repr import SchemaRepr


class TestDAGRepr(unittest.TestCase):
    """
    DAGRepr test
    """
    dag_dict = {
        '_input': {
            'inputs': None,
            'outputs': ('a', 'b', 'c'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'op_info': {
                'operator': 'NOPNodeOperator',
                'type': 'built_in',
                'init_args': None,
                'init_kws': None,
                'tag': None
            },
            'next_nodes': ['op1']
        },
        'op1': {
            'inputs': ('a', 'c'),
            'outputs': ('a', 'c'),
            'iter_info': {
                'type': 'flat_map',
                'param': None
            },
            'op_info': {
                'operator': 'towhee/op1',
                'type': 'hub',
                'init_args': ('x',),
                'init_kws': {'y': 'y'},
                'tag': 'main',
            },
            'config': None,
            'next_nodes': ['op2']
        },
        'op2': {
            'inputs': ('a', 'b'),
            'outputs': ('d', 'e'),
            'iter_info': {
                'type': 'filter',
                'param': {'filter_by': 'a'}
            },
            'op_info': {
                'operator': 'towhee/op2',
                'type': 'hub',
                'init_args': ('a',),
                'init_kws': {'b': 'b'},
                'tag': '1.1',
            },
            'config': {'parallel': 3},
            'next_nodes': ['_output']
        },
        '_output': {
            'inputs': ('d', 'c'),
            'outputs': ('d', 'c'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'op_info': {
                'operator': 'NOPNodeOperator',
                'type': 'built_in',
                'init_args': None,
                'init_kws': None,
                'tag': None
            },
            'next_nodes': None
        },
    }

    def test_dag(self):
        dr = DAGRepr.from_dict(self.dag_dict)
        edges = dr.edges
        nodes = dr.nodes
        dag_dict = dr.dag_dict
        top_sort = dr.top_sort
        self.assertEqual(len(edges), 5)
        self.assertEqual(len(nodes), 4)
        self.assertEqual(len(dag_dict), 4)
        self.assertEqual(len(top_sort), 4)
        for edge in edges.values():
            for schema in edge['schema']:
                self.assertTrue(isinstance(edge['schema'][schema], SchemaRepr))
        for node in nodes:
            self.assertTrue(isinstance(dr.nodes[node], NodeRepr))

        dr = DAGRepr(nodes, edges)
        edges = dr.edges
        nodes = dr.nodes
        dag_dict = dr.dag_dict
        top_sort = dr.top_sort
        self.assertEqual(len(edges), 5)
        self.assertEqual(len(nodes), 4)
        self.assertEqual(dag_dict, None)
        self.assertEqual(len(top_sort), 4)

    def test_check_input(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test.pop('_input')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_output(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test.pop('_output')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_schema(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['op1']['inputs'] = ('x', 'y')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_schema_circle(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['op2']['next_nodes'] = ['op1', '_output']
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_edges(self):
        """
        _input(map)[(a, b, c)]->op1(flat_map)[(a, c)-(a, c)]->op2(filter)[(a, b)-(d, e)]->_output(map)[(d, c)]
        """
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        dr = DAGRepr.from_dict(towhee_dag_test)
        edges = dr.edges
        nodes = dr.nodes
        self.assertEqual(len(edges), 5)
        self.assertEqual(len(nodes), 4)

        edge0 = [('a', ColumnType.SCALAR), ('b', ColumnType.SCALAR), ('c', ColumnType.SCALAR)]
        edge1 = [('a', ColumnType.QUEUE), ('c', ColumnType.QUEUE), ('b', ColumnType.SCALAR)]
        edge2 = [('d', ColumnType.QUEUE), ('c', ColumnType.QUEUE)]
        self.assertEqual(dict((s, t) for s, t in edges[0]['data']), dict((s, t) for s, t in edge0))
        self.assertEqual(dict((s, t) for s, t in edges[1]['data']), dict((s, t) for s, t in edge0))
        self.assertEqual(dict((s, t) for s, t in edges[2]['data']), dict((s, t) for s, t in edge1))
        self.assertEqual(dict((s, t) for s, t in edges[3]['data']), dict((s, t) for s, t in edge2))
        self.assertEqual(dict((s, t) for s, t in edges[4]['data']), dict((s, t) for s, t in edge2))

        self.assertEqual(nodes['_input'].in_edges, [0])
        self.assertEqual(nodes['_input'].out_edges, [1])
        self.assertEqual(nodes['op1'].in_edges, [1])
        self.assertEqual(nodes['op1'].out_edges, [2])
        self.assertEqual(nodes['op2'].in_edges, [2])
        self.assertEqual(nodes['op2'].out_edges, [3])
        self.assertEqual(nodes['_output'].in_edges, [3])
        self.assertEqual(nodes['_output'].out_edges, [4])

    def test_concat(self):
        """
        _input[(a,b)]->op1[(a,)-(c,)]->op3[(c, d)-(c, d)]->_output[(c, d)]
                |------>op2[(b,)-(d,)]----^
        """
        towhee_dag_test = {
            '_input': {
                'inputs': ('a', 'b'),
                'outputs': ('a', 'b'),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'NOPNodeOperator', 'type': 'built_in', 'init_args': None, 'init_kws': None, 'tag': None},
                'next_nodes': ['op1', 'op2']
            },
            'op1': {
                'inputs': ('a',),
                'outputs': ('c',),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'test1', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['op3']
            },
            'op2': {
                'inputs': ('b',),
                'outputs': ('d',),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'test2', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['op3']
            },
            'op3': {
                'inputs': (),
                'outputs': (),
                'iter_info': {'type': 'concat', 'param': None},
                'op_info': {'operator': 'test3', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['_output']
            },
            '_output': {
                'inputs': ('c', 'd'),
                'outputs': ('c', 'd'),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'NOPNodeOperator', 'type': 'built_in', 'init_args': None, 'init_kws': None, 'tag': None},
                'next_nodes': None
            },
        }
        dr = DAGRepr.from_dict(towhee_dag_test)
        edges = dr.edges
        nodes = dr.nodes
        self.assertEqual(len(edges), 7)
        self.assertEqual(len(nodes), 5)

        edge0 = [('a', ColumnType.SCALAR), ('b', ColumnType.SCALAR)]
        edge1 = [('a', ColumnType.SCALAR)]
        edge2 = [('b', ColumnType.SCALAR)]
        edge3 = [('c', ColumnType.SCALAR)]
        edge4 = [('d', ColumnType.SCALAR)]
        edge5 = [('c', ColumnType.SCALAR), ('d', ColumnType.SCALAR)]
        self.assertEqual(dict((s, t) for s, t in edges[0]['data']), dict((s, t) for s, t in edge0))
        self.assertEqual(dict((s, t) for s, t in edges[1]['data']), dict((s, t) for s, t in edge1))
        self.assertEqual(dict((s, t) for s, t in edges[2]['data']), dict((s, t) for s, t in edge2))
        self.assertEqual(dict((s, t) for s, t in edges[3]['data']), dict((s, t) for s, t in edge3))
        self.assertEqual(dict((s, t) for s, t in edges[4]['data']), dict((s, t) for s, t in edge4))
        self.assertEqual(dict((s, t) for s, t in edges[5]['data']), dict((s, t) for s, t in edge5))
        self.assertEqual(dict((s, t) for s, t in edges[6]['data']), dict((s, t) for s, t in edge5))

        self.assertEqual(nodes['_input'].in_edges, [0])
        self.assertEqual(nodes['_input'].out_edges, [1, 2])
        self.assertEqual(nodes['op1'].in_edges, [1])
        self.assertEqual(nodes['op1'].out_edges, [3])
        self.assertEqual(nodes['op2'].in_edges, [2])
        self.assertEqual(nodes['op2'].out_edges, [4])
        self.assertEqual(nodes['op3'].in_edges, [3, 4])
        self.assertEqual(nodes['op3'].out_edges, [5])
        self.assertEqual(nodes['_output'].in_edges, [5])
        self.assertEqual(nodes['_output'].out_edges, [6])

    def test_concat1(self):
        """
        _input[(a,)]->op1[(a,)-(b, c)]->op2[(c,)-(c,)]->op3[(b,)-(e,)]->_output[(f, e)]
                                                      |-->op4[(a,)-(f,)]--^
        """
        towhee_dag_test = {
            '_input': {
                'inputs': ('a',),
                'outputs': ('a',),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'NOPNodeOperator', 'type': 'built_in', 'init_args': None, 'init_kws': None, 'tag': None},
                'next_nodes': ['op1']
            },
            'op1': {
                'inputs': ('a',),
                'outputs': ('b', 'c'),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'test1', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['op2']
            },
            'op2': {
                'inputs': ('c',),
                'outputs': ('c',),
                'iter_info': {'type': 'flat_map', 'param': None},
                'op_info': {'operator': 'test2', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['op3', 'op4']
            },
            'op3': {
                'inputs': ('b',),
                'outputs': ('e',),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'test3', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['op5']
            },
            'op4': {
                'inputs': ('a',),
                'outputs': ('f',),
                'iter_info': {'type': 'flat_map', 'param': None},
                'op_info': {'operator': 'test4', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['op5']
            },
            'op5': {
                'inputs': (),
                'outputs': (),
                'iter_info': {'type': 'concat', 'param': None},
                'op_info': {'operator': 'test5', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['_output']
            },
            '_output': {
                'inputs': ('e', 'f'),
                'outputs': ('e', 'f'),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'NOPNodeOperator', 'type': 'built_in', 'init_args': None, 'init_kws': None, 'tag': None},
                'next_nodes': None
            },
        }
        dr = DAGRepr.from_dict(towhee_dag_test)
        edges = dr.edges
        nodes = dr.nodes
        self.assertEqual(len(edges), 9)
        self.assertEqual(len(nodes), 7)

        edge0 = [('a', ColumnType.SCALAR)]
        edge1 = [('a', ColumnType.SCALAR)]
        edge2 = [('a', ColumnType.SCALAR), ('b', ColumnType.SCALAR), ('c', ColumnType.SCALAR)]
        edge3 = [('b', ColumnType.SCALAR)]
        edge4 = [('a', ColumnType.SCALAR)]
        edge5 = [('e', ColumnType.SCALAR)]
        edge6 = [('f', ColumnType.QUEUE)]
        edge7 = [('e', ColumnType.SCALAR), ('f', ColumnType.QUEUE)]
        edge8 = [('e', ColumnType.SCALAR), ('f', ColumnType.QUEUE)]
        self.assertEqual(dict((s, t) for s, t in edges[0]['data']), dict((s, t) for s, t in edge0))
        self.assertEqual(dict((s, t) for s, t in edges[1]['data']), dict((s, t) for s, t in edge1))
        self.assertEqual(dict((s, t) for s, t in edges[2]['data']), dict((s, t) for s, t in edge2))
        self.assertEqual(dict((s, t) for s, t in edges[3]['data']), dict((s, t) for s, t in edge3))
        self.assertEqual(dict((s, t) for s, t in edges[4]['data']), dict((s, t) for s, t in edge4))
        self.assertEqual(dict((s, t) for s, t in edges[5]['data']), dict((s, t) for s, t in edge5))
        self.assertEqual(dict((s, t) for s, t in edges[6]['data']), dict((s, t) for s, t in edge6))
        self.assertEqual(dict((s, t) for s, t in edges[7]['data']), dict((s, t) for s, t in edge7))
        self.assertEqual(dict((s, t) for s, t in edges[8]['data']), dict((s, t) for s, t in edge8))

        self.assertEqual(nodes['_input'].in_edges, [0])
        self.assertEqual(nodes['_input'].out_edges, [1])
        self.assertEqual(nodes['op1'].in_edges, [1])
        self.assertEqual(nodes['op1'].out_edges, [2])
        self.assertEqual(nodes['op2'].in_edges, [2])
        self.assertEqual(nodes['op2'].out_edges, [3, 4])
        self.assertEqual(nodes['op3'].in_edges, [3])
        self.assertEqual(nodes['op3'].out_edges, [5])
        self.assertEqual(nodes['op4'].in_edges, [4])
        self.assertEqual(nodes['op4'].out_edges, [6])
        self.assertEqual(nodes['op5'].in_edges, [5, 6])
        self.assertEqual(nodes['op5'].out_edges, [7])
        self.assertEqual(nodes['_output'].in_edges, [7])
        self.assertEqual(nodes['_output'].out_edges, [8])

    def test_filter(self):
        """
        _input[(u,)]->op1[(u,)-(i,)]->op2[(i,)-(b, c, s)]->op3[(i, b)-(i, b), filter:(b, )]->_output[(c, s, b)]
        """
        towhee_dag_test = {
            '_input': {
                'inputs': ('u',),
                'outputs': ('u',),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'NOPNodeOperator', 'type': 'built_in', 'init_args': None, 'init_kws': None, 'tag': None},
                'next_nodes': ['op1']
            },
            'op1': {
                'inputs': ('u',),
                'outputs': ('i',),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'test1', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['op2']
            },
            'op2': {
                'inputs': ('i',),
                'outputs': ('b', 'c', 's'),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'test2', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['op3']
            },
            'op3': {
                'inputs': ('i', 'b'),
                'outputs': ('i', 'b'),
                'iter_info': {'type': 'filter', 'param': {'filter_by': ('b',)}},
                'op_info': {'operator': 'test3', 'type': 'hub', 'init_args': None, 'init_kws': None, 'tag': 'main'},
                'config': None,
                'next_nodes': ['_output']
            },
            '_output': {
                'inputs': ('c', 's', 'b'),
                'outputs': ('c', 's', 'b'),
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'NOPNodeOperator', 'type': 'built_in', 'init_args': None, 'init_kws': None, 'tag': None},
                'next_nodes': None
            },
        }
        dr = DAGRepr.from_dict(towhee_dag_test)
        edges = dr.edges
        nodes = dr.nodes
        self.assertEqual(len(edges), 6)
        self.assertEqual(len(nodes), 5)

        edge0 = [('u', ColumnType.SCALAR)]
        edge1 = [('u', ColumnType.SCALAR)]
        edge2 = [('i', ColumnType.SCALAR)]
        edge3 = [('i', ColumnType.SCALAR), ('b', ColumnType.SCALAR), ('c', ColumnType.SCALAR), ('s', ColumnType.SCALAR)]
        edge4 = [('b', ColumnType.SCALAR), ('c', ColumnType.SCALAR), ('s', ColumnType.SCALAR)]
        edge5 = [('b', ColumnType.SCALAR), ('c', ColumnType.SCALAR), ('s', ColumnType.SCALAR)]
        self.assertEqual(dict((s, t) for s, t in edges[0]['data']), dict((s, t) for s, t in edge0))
        self.assertEqual(dict((s, t) for s, t in edges[1]['data']), dict((s, t) for s, t in edge1))
        self.assertEqual(dict((s, t) for s, t in edges[2]['data']), dict((s, t) for s, t in edge2))
        self.assertEqual(dict((s, t) for s, t in edges[3]['data']), dict((s, t) for s, t in edge3))
        self.assertEqual(dict((s, t) for s, t in edges[4]['data']), dict((s, t) for s, t in edge4))
        self.assertEqual(dict((s, t) for s, t in edges[5]['data']), dict((s, t) for s, t in edge5))

        self.assertEqual(nodes['_input'].in_edges, [0])
        self.assertEqual(nodes['_input'].out_edges, [1])
        self.assertEqual(nodes['op1'].in_edges, [1])
        self.assertEqual(nodes['op1'].out_edges, [2])
        self.assertEqual(nodes['op2'].in_edges, [2])
        self.assertEqual(nodes['op2'].out_edges, [3])
        self.assertEqual(nodes['op3'].in_edges, [3])
        self.assertEqual(nodes['op3'].out_edges, [4])
        self.assertEqual(nodes['_output'].in_edges, [4])
        self.assertEqual(nodes['_output'].out_edges, [5])

    def test_to_json(self):

        def add(x):
            return x + 1

        p = (
            pipe.input('a')
                .flat_map('a', 'b', lambda x: x)
                .map('a', 'b', add)
                .output('a', 'b')
        )

        edges = {
            0: [{'name': 'a', 'type': 'SCALAR'}],
            1: [{'name': 'a', 'type': 'SCALAR'}],
            2: [{'name': 'a', 'type': 'SCALAR'}],
            3: [{'name': 'a', 'type': 'SCALAR'}, {'name': 'b', 'type': 'SCALAR'}],
            4: [{'name': 'a', 'type': 'SCALAR'}, {'name': 'b', 'type': 'SCALAR'}]
        }

        nodes = {
            '_input': {
                'name': '_input',
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'NOPNodeOperator', 'type': 'built_in'},
                'next_nodes': ['mock_a'],
                'inputs': [0],
                'outputs': [1],
                'op_input': ('a',),
                'op_output': ('a',)
            },
            'mock_a': {
                'name': 'lambda-0',
                'iter_info': {'type': 'flat_map', 'param': None},
                'op_info': {'operator': '<lambda>', 'type': 'lambda'},
                'next_nodes': ['mock_b'],
                'inputs': [1],
                'outputs': [2],
                'op_input': ('a',),
                'op_output': ('b',)
            },
            'mock_b': {
                'name': 'add-1',
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'add', 'type': 'callable'},
                'next_nodes': ['_output'],
                'inputs': [2],
                'outputs': [3],
                'op_input': ('a',),
                'op_output': ('b',)
            },
            '_output': {
                'name': '_output',
                'iter_info': {'type': 'map', 'param': None},
                'op_info': {'operator': 'NOPNodeOperator', 'type': 'built_in'},
                'next_nodes': [],
                'inputs': [3],
                'outputs': [4],
                'op_input': ('a', 'b'),
                'op_output': ('a', 'b')
            }
        }

        for v1, v2 in zip(nodes.values(), p.dag_repr.to_dict()['nodes'].values()):
            self.assertEqual(v1['name'], v2['name'])
            self.assertEqual(v1['iter_info'], v2['iter_info'])
            self.assertEqual(v1['inputs'], v2['inputs'])
            self.assertEqual(v1['outputs'], v2['outputs'])
            self.assertEqual(v1['op_input'], v2['op_input'])
            self.assertEqual(v1['op_output'], v2['op_output'])

        for v1, v2 in zip(edges.values(), p.dag_repr.to_dict()['edges'].values()):
            for i in v1:
                self.assertIn(i, v2)

        p.dag_repr.to_json()
