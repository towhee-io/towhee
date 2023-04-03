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
import unittest

from towhee import pipe
from towhee.tools import visualizer
from towhee.tools.visualizer import Visualizer
from towhee.runtime.data_queue import DataQueue
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


class TestVisualizer(unittest.TestCase):
    """
    Unit tests for Visualizer.
    """
    def test_dag_visualizer(self):
        p = _get_pipe()
        visualizer.show_graph(p)

    def test_show_tracer(self):
        p = _get_pipe()
        v = Visualizer(
            nodes=p.dag_repr.to_dict().get('nodes'),
            data_queues=[p.preload().data_queues],
            trace_nodes=[i.name for i in p.dag_repr.nodes.values()]
        )
        v.tracer.show()

    def test_get_tracer(self):
        p = _get_pipe()
        v = Visualizer(
            nodes=p.dag_repr.to_dict().get('nodes'),
            data_queues=[p.preload().data_queues],
            trace_nodes=[i.name for i in p.dag_repr.nodes.values()]
        )
        t = v.tracer[0]
        t['_input'].show()
        self.assertEqual(t['_input'].name, '_input')
        self.assertEqual(t['_input'].op_input, ('a',))
        t['lambda-0'].show()
        self.assertEqual(t['lambda-0'].name, 'lambda-0')
        self.assertEqual(t['lambda-0'].op_input, ('a',))
        t['lambda-1'].show()
        self.assertEqual(t['lambda-1'].name, 'lambda-1')
        self.assertEqual(t['lambda-1'].op_input, ('b', ))
        t['lambda-2'].show()
        self.assertEqual(t['lambda-2'].name, 'lambda-2')
        self.assertEqual(t['lambda-2'].op_input, ('b',))
        t['lambda-3'].show()
        self.assertEqual(t['lambda-3'].name, 'lambda-3')
        self.assertEqual(t['lambda-3'].op_input, ('b',))
        t['concat-4'].show()
        self.assertEqual(t['concat-4'].name, 'concat-4')
        self.assertEqual(t['concat-4'].op_input, ('e', 'd', 'c'))
        t['_output'].show()
        self.assertEqual(t['_output'].name, '_output')
        self.assertEqual(t['_output'].op_input, ('b', 'c', 'd', 'e'))
        t.show()

    def test_false_node(self):
        p = _get_pipe()
        v = Visualizer(
            nodes=p.dag_repr.to_dict().get('nodes'),
            data_queues=[p.preload().data_queues],
            trace_nodes=[i.name for i in p.dag_repr.nodes.values()]
        )
        t = v.tracer[0]
        with self.assertRaises(KeyError):
            _ = t['test']

    def test_result(self):
        p = (
            pipe.input('a')
                .map('a', 'b', lambda x: x + 1)
                .output('a', 'b')
        )

        v = p.debug(1, tracer=True)
        res = v.result.get()

        self.assertIsInstance(v.result, DataQueue)
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1], 2)

    def test_profiler(self):
        p = (
            pipe.input('a')
                .map('a', 'b', lambda x: x + 1)
                .output('a', 'b')
        )

        _ = p.debug(1, profiler=True)
        _ = p.debug(2, profiler=True)
        v = p.debug(3, profiler=True)
        pp = v.profiler

        self.assertEqual(len(pp._time_prfilers), 1)

        pp.show()

    def test_no_profiler(self):
        p = (
            pipe.input('a')
                .map('a', 'b', lambda x: x + 1)
                .output('a', 'b')
        )
        v = p.debug(3, tracer=True)
        self.assertIsNone(v.profiler)

    def test_tracer(self):
        p = (
            pipe.input('a')
                .map('a', 'b', lambda x: x + 1)
                .map('a', 'b', lambda x: x + 2)
                .output('a', 'b')
        )
        v = p.debug(3, tracer=True)
        v.tracer.show()
        t = v.tracer[0]
        self.assertEqual(len(t['lambda']), 2)

        self.assertEqual(t['_input'].next_node, ['lambda-0'])
        self.assertEqual(t['lambda-0'].next_node, ['lambda-1'])
        self.assertEqual(t['lambda-0'].previous_node, ['_input'])
        self.assertEqual(t['lambda-1'].next_node, ['_output'])
        self.assertEqual(t['lambda-1'].previous_node, ['lambda-0'])

    def test_json(self):
        p = (
            pipe.input('a')
                .map('a', 'b', lambda x: x + 1)
                .output('a', 'b')
        )

        v = p.debug(1, profiler=True)
        v._result = None
        info = v.to_json()

        v1 =  Visualizer.from_json(info)
        self.assertListEqual([i.time_record for i in v.time_profiler], [i.time_record for i in v1.time_profiler])
