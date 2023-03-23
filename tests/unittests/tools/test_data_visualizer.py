# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file excepv in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from towhee import pipe
from towhee.tools.data_visualizer import PipeVisualizer
from towhee.runtime.data_queue import DataQueue, ColumnType
# pylint: disable=protected-access


def _get_pipe():
    p = (
        pipe.input('a')
            .flat_map('a', 'b', lambda x: x)
    )

    p1 = (
        p.map('b', 'c', lambda x: x + 1)
    )

    p2 = (
        p.map('b', 'd', lambda x: x + 2)
    )

    p3 = (
        p.map('b', 'e', lambda x: x + 3)
    )

    p4 = (
        p1.concat(p2, p3).output('b', 'c', 'd', 'e')
    )

    return p4


def _get_node_queue():
    q1 = DataQueue([('a', ColumnType.SCALAR)])
    q1.put(([1,2],))
    q1_1 = DataQueue([('a', ColumnType.SCALAR)])
    q1_1.put(([1,2],))
    q1_2 = DataQueue([('a', ColumnType.SCALAR)])
    q1_2.put(([1,2],))

    q2 = DataQueue([('b', ColumnType.QUEUE)])
    q2.put((1,))
    q2.put((2,))
    q2_1 = DataQueue([('b', ColumnType.QUEUE)])
    q2_1.put((1,))
    q2_1.put((2,))
    q2_2 = DataQueue([('b', ColumnType.QUEUE)])
    q2_2.put((1,))
    q2_2.put((2,))
    q2_3 = DataQueue([('b', ColumnType.QUEUE)])
    q2_3.put((1,))
    q2_3.put((2,))

    q3 = DataQueue([('b', ColumnType.QUEUE), ('c', ColumnType.QUEUE)])
    q3.put((1, 2))
    q3.put((2, 3))

    q4 = DataQueue([('b', ColumnType.QUEUE), ('d', ColumnType.QUEUE)])
    q4.put((1, 3))
    q4.put((2, 4))

    q5 = DataQueue([('b', ColumnType.QUEUE), ('e', ColumnType.QUEUE)])
    q5.put((1, 4))
    q5.put((2, 5))

    q6 = DataQueue([('c', ColumnType.QUEUE)])
    q6.put((2,))
    q6.put((3,))

    q7 = DataQueue([('d', ColumnType.QUEUE)])
    q7.put((3,))
    q7.put((4,))

    q8 = DataQueue([('b', ColumnType.QUEUE), ('e', ColumnType.QUEUE)])
    q8.put((1, 4))
    q8.put((2, 5))

    q9 = DataQueue([('b', ColumnType.QUEUE), ('c', ColumnType.QUEUE), ('d', ColumnType.QUEUE), ('e', ColumnType.QUEUE)])
    q9.put((1, 2, 3, 4))
    q9.put((2, 3, 4, 5))

    q9_1 = DataQueue([('b', ColumnType.QUEUE), ('c', ColumnType.QUEUE), ('d', ColumnType.QUEUE), ('e', ColumnType.QUEUE)])
    q9_1.put((1, 2, 3, 4))
    q9_1.put((2, 3, 4, 5))

    q9_2 = DataQueue([('b', ColumnType.QUEUE), ('c', ColumnType.QUEUE), ('d', ColumnType.QUEUE), ('e', ColumnType.QUEUE)])
    q9_2.put((1, 2, 3, 4))
    q9_2.put((2, 3, 4, 5))

    q = {
        '_input': {'in': q1,
                    'op_input': ['a'],
                    'out': q1_1},
        'lambda-0': {'in': q1_2,
                    'op_input': ['a'],
                    'out': q2},
        'lambda-1': {'in': q2_1,
                    'op_input': ['b'],
                    'out': q3},
        'lambda-2': {'in': q2_2,
                    'op_input': ['b'],
                    'out': q4},
        'lambda-3': {'in': q2_3,
                    'op_input': ['b'],
                    'out': q5},
        'concat-4': {'in': [q6, q7, q8],
                    'op_input': None,
                    'out': q9},
        '_output': {'in': q9_1,
                    'op_input': ['b', 'c', 'd', 'e'],
                    'out': q9_2}
    }

    return q


class TestDataVisualizer(unittest.TestCase):
    """
    Unit test for data visualizers.
    """
    def test_visualizer(self):
        p = _get_pipe()
        q = _get_node_queue()
        pv = PipeVisualizer(p.dag_repr.to_dict().get('nodes'), q)
        pv['_input'].show()
        self.assertEqual(pv['_input'].name, '_input')
        self.assertIsNone(pv['_input'].previous_node)
        self.assertEqual(pv['_input'].op_input, ['a'])
        self.assertEqual(pv['_input'].inputs.to_list()[0]['a'], [1,2])
        self.assertEqual(pv['_input'].outputs.to_list()[0]['a'], [1,2])
        pv['lambda-0'].show()
        self.assertEqual(pv['lambda-0'].name, 'lambda-0')
        self.assertEqual(pv['lambda-0'].previous_node, ['_input'])
        self.assertEqual(pv['lambda-0'].op_input, ['a'])
        self.assertEqual(pv['lambda-0'].inputs.to_list()[0]['a'], [1,2])
        self.assertEqual(pv['lambda-0'].outputs.to_list()[0]['b'], 1)
        self.assertEqual(pv['lambda-0'].outputs.to_list()[1]['b'], 2)
        pv['lambda-1'].show()
        self.assertEqual(pv['lambda-1'].name, 'lambda-1')
        self.assertEqual(pv['lambda-1'].previous_node, ['lambda-0'])
        self.assertEqual(pv['lambda-1'].op_input, ['b'])
        self.assertEqual(pv['lambda-1'].inputs.to_list()[0]['b'], 1)
        self.assertEqual(pv['lambda-1'].inputs.to_list()[1]['b'], 2)
        self.assertEqual(pv['lambda-1'].outputs.to_list()[0]['c'], 2)
        self.assertEqual(pv['lambda-1'].outputs.to_list()[1]['c'], 3)
        pv['lambda-2'].show()
        self.assertEqual(pv['lambda-2'].name, 'lambda-2')
        self.assertEqual(pv['lambda-2'].previous_node, ['lambda-0'])
        self.assertEqual(pv['lambda-2'].op_input, ['b'])
        self.assertEqual(pv['lambda-2'].inputs.to_list()[0]['b'], 1)
        self.assertEqual(pv['lambda-2'].inputs.to_list()[1]['b'], 2)
        self.assertEqual(pv['lambda-2'].outputs.to_list()[0]['d'], 3)
        self.assertEqual(pv['lambda-2'].outputs.to_list()[1]['d'], 4)
        pv['lambda-3'].show()
        self.assertEqual(pv['lambda-3'].name, 'lambda-3')
        self.assertEqual(pv['lambda-3'].previous_node, ['lambda-0'])
        self.assertEqual(pv['lambda-3'].op_input, ['b'])
        self.assertEqual(pv['lambda-3'].inputs.to_list()[0]['b'], 1)
        self.assertEqual(pv['lambda-3'].inputs.to_list()[1]['b'], 2)
        self.assertEqual(pv['lambda-3'].outputs.to_list()[0]['e'], 4)
        self.assertEqual(pv['lambda-3'].outputs.to_list()[1]['e'], 5)
        pv['concat-4'].show()
        self.assertEqual(pv['concat-4'].name, 'concat-4')
        self.assertEqual(pv['concat-4'].previous_node, ['lambda-1', 'lambda-2', 'lambda-3'])
        self.assertIsNone(pv['concat-4'].op_input)
        self.assertEqual(pv['concat-4'].inputs[2].to_list()[0]['b'], 1)
        self.assertEqual(pv['concat-4'].inputs[2].to_list()[1]['b'], 2)
        self.assertEqual(pv['concat-4'].outputs.to_list()[0]['c'], 2)
        self.assertEqual(pv['concat-4'].outputs.to_list()[1]['c'], 3)
        self.assertEqual(pv['concat-4'].outputs.to_list()[0]['d'], 3)
        self.assertEqual(pv['concat-4'].outputs.to_list()[1]['d'], 4)
        self.assertEqual(pv['concat-4'].outputs.to_list()[0]['e'], 4)
        self.assertEqual(pv['concat-4'].outputs.to_list()[1]['e'], 5)
        pv.show()

    def test_false_node(self):
        p = _get_pipe()
        q = _get_node_queue()
        pv = PipeVisualizer(p.dag_repr.to_dict().get('nodes'), q)
        with self.assertRaises(KeyError):
            _ = pv['test']
