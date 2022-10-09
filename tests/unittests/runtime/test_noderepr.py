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
            'fn_type': '_input',
            'iteration': 'map'
        }
        node = NodeRepr.from_dict('_input', node_input)
        self.assertEqual(node.name, '_input')
        self.assertEqual(node.inputs, ('a', 'b'))
        self.assertEqual(node.outputs, ('a', 'b'))
        self.assertEqual(node.fn_type, '_input')
        self.assertEqual(node.iteration, 'map')

    def test_op(self):
        node_op = {
            'function': 'towhee.test',
            'init_args': ('a',),
            'init_kws': {'b': 'b'},
            'inputs': ('a', 'b'),
            'outputs': 'd',
            'fn_type': 'hub',
            'iteration': 'filter',
            'config': {'parallel': 3},
            'tag': '1.1',
            'param': {'filter_columns': 'a'}
        }
        node = NodeRepr.from_dict('test_dict', node_op)
        self.assertEqual(node.name, 'test_dict')
        self.assertEqual(node.function, 'towhee.test')
        self.assertEqual(node.init_args, ('a',))
        self.assertEqual(node.init_kws, {'b': 'b'})
        self.assertEqual(node.inputs, ('a', 'b'))
        self.assertEqual(node.outputs, 'd')
        self.assertEqual(node.fn_type, 'hub')
        self.assertEqual(node.iteration, 'filter')
        self.assertEqual(node.config, {'parallel': 3})
        self.assertEqual(node.tag, '1.1')
        self.assertEqual(node.param, {'filter_columns': 'a'})

    def test_raise(self):
        node_input = {
            'inputs': ('a', 'b'),
            'outputs': ('a', 'b'),
            'fn_type': '_input',
        }
        with self.assertRaises(ValueError):
            NodeRepr.from_dict('_input', node_input)
