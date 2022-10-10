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

import unittest

from towhee.runtime.dag_repr import DAGRepr, NodeRepr


class TestDAGRepr(unittest.TestCase):
    """
    DAGRepr test
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.towhee_dag = {
            '_input': {
                'inputs': ('a', 'b'),
                'outputs': ('a', 'b'),
                'iter_info': {
                    'type': 'map',
                    'param': None
                }
            },
            'e433a': {
                'inputs': ('a',),
                'outputs': ('c',),
                'iter_info': {
                    'type': 'map',
                    'param': None
                },
                'op_info': {
                    'operator': 'towhee.decode',
                    'type': 'hub',
                    'init_args': ('a',),
                    'init_kws': {'b': 'b'},
                    'tag': 'main',
                },
                'config': None,
            },
            'b1196': {
                'inputs': ('a', 'b'),
                'outputs': ('d',),
                'iter_info': {
                    'type': 'filter',
                    'param': {'filter_columns': 'a'}
                },
                'op_info': {
                    'operator': 'towhee.test',
                    'type': 'hub',
                    'init_args': ('a',),
                    'init_kws': {'b': 'b'},
                    'tag': '1.1',
                },
                'config': {'parallel': 3},
            },
            '_output': {
                'inputs': ('d',),
                'outputs': ('d',),
                'iter_info': {
                    'type': 'map',
                    'param': None
                }
            },
        }

    def test_dag(self):
        dr = DAGRepr.from_dict(self.towhee_dag)
        self.assertEqual(dr.dag_type, 'local')

        node1 = dr.nodes
        node2 = node1.next_node
        node3 = node2.next_node
        node4 = node3.next_node
        self.assertTrue(isinstance(node1, NodeRepr))
        self.assertTrue(isinstance(node2, NodeRepr))
        self.assertTrue(isinstance(node3, NodeRepr))
        self.assertTrue(isinstance(node4, NodeRepr))

    def test_check_input(self):
        towhee_dag_test = self.towhee_dag
        towhee_dag_test.pop('_input')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_output(self):
        towhee_dag_test = self.towhee_dag
        towhee_dag_test.pop('_output')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_schema(self):
        towhee_dag_test = self.towhee_dag
        towhee_dag_test['b1196']['inputs'] = ('x', 'y')
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)

    def test_check_schema_equal(self):
        towhee_dag_test = self.towhee_dag
        towhee_dag_test['_input']['inputs'] = ('x',)
        with self.assertRaises(ValueError):
            DAGRepr.from_dict(towhee_dag_test)
