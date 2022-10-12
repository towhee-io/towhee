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

from towhee.runtime.node_repr import NodeRepr


class TestNodeRepr(unittest.TestCase):
    """
    NodeRepr test
    """
    def test_input(self):
        node_input = {
            'inputs': ('a', 'b'),
            'outputs': ('a', 'b'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': ['next']
        }
        node = NodeRepr.from_dict('_input', node_input)
        self.assertEqual(node.name, '_input')
        self.assertEqual(node.inputs, ('a', 'b'))
        self.assertEqual(node.outputs, ('a', 'b'))
        self.assertEqual(node.iter_info.type, 'map')
        self.assertEqual(node.iter_info.param, None)
        self.assertEqual(node.next_nodes, ['next'])

    def test_op(self):
        node_op = {
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
            'next_nodes': ['next1', 'next2']
        }
        node = NodeRepr.from_dict('test_dict', node_op)
        self.assertEqual(node.name, 'test_dict')
        self.assertEqual(node.inputs, ('a', 'b'))
        self.assertEqual(node.outputs, ('d',))
        self.assertEqual(node.iter_info.type, 'filter')
        self.assertEqual(node.iter_info.param, {'filter_columns': 'a'})
        self.assertEqual(node.op_info.operator, 'towhee.test')
        self.assertEqual(node.op_info.type, 'hub')
        self.assertEqual(node.op_info.init_args, ('a', ))
        self.assertEqual(node.op_info.init_kws, {'b': 'b'})
        self.assertEqual(node.op_info.tag, '1.1')
        self.assertEqual(node.config, {'parallel': 3})
        self.assertEqual(node.next_nodes, ['next1', 'next2'])

    def test_raise_iter(self):
        node_input = {
            'inputs': ('a', 'b'),
            'outputs': ('a', 'b'),
        }
        with self.assertRaises(ValueError):
            NodeRepr.from_dict('_input', node_input)

    def test_raise_op(self):
        node_input = {
            'inputs': ('a', 'b'),
            'outputs': ('a', 'b'),
            'iter_info': {
                'type': 'map',
                'param': None
            }
        }
        with self.assertRaises(ValueError):
            NodeRepr.from_dict('test_op', node_input)
