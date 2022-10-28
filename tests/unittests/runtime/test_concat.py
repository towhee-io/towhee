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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.nodes import create_node, NodeStatus
from towhee.runtime.data_queue import DataQueue, ColumnType
from towhee.runtime.operator_manager import OperatorPool


class TestMapNode(unittest.TestCase):
    """
    map node test.
    """
    node_info = {
        'inputs': ('num', ),
        'outputs': ('vec', ),
        'op_info': {
            'type': 'local',
            'operator': '_concat',
            'tag': 'main',
            'init_args': None,
            'init_kws': None
        },
        'iter_info': {
            'type': 'concat',
            'param': None
        },
        'config': {},
        'next_nodes': ['_output']
    }


    def setUp(self):
        self.node_repr = NodeRepr.from_dict(uuid.uuid4().hex, self.node_info)
        self.op_pool = OperatorPool()
        self.thread_pool = ThreadPoolExecutor()

    def test_normal(self):
        in_que1 = DataQueue([('url', ColumnType.SCALAR), ('num1', ColumnType.QUEUE)])
        in_que2 = DataQueue([('url', ColumnType.SCALAR), ('num2', ColumnType.QUEUE)])
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num1', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que1, in_que2], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)

        in_que1.put(('test', 1))
        in_que1.put(('test', 2))
        in_que1.put(('test', 3))

        in_que2.put(('test', 2))
        in_que2.put(('test', 3))
        in_que2.put(('test', 4))
        in_que1.seal()
        in_que2.seal()
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        exp = [
            {'url': 'test', 'num1': 1, 'num2': 2},
            {'url': 'test', 'num1': 2, 'num2': 3},
            {'url': 'test', 'num1': 3, 'num2': 4}
        ]

        ret = []
        while out_que1.size > 0:
            ret.append(out_que1.get_dict())
        self.assertEqual(ret, exp)

        exp = [
            {'num1': 1},
            {'num1': 2},
            {'num1': 3}
        ]

        ret = []
        while out_que2.size > 0:
            ret.append(out_que2.get_dict())
        self.assertEqual(ret, exp)

    def test_multithread(self):
        in_que1 = DataQueue([('url', ColumnType.SCALAR), ('num1', ColumnType.QUEUE)])
        in_que2 = DataQueue([('url', ColumnType.SCALAR), ('num2', ColumnType.QUEUE)])
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num1', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que1, in_que2], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)

        def write(size):
            for i in range(size):
                in_que1.put(('test_url', i))
                time.sleep(0.01)
                in_que2.put(('test_url', i + 1))
            in_que1.seal()
            in_que2.seal()

        size = 10
        w_f = self.thread_pool.submit(write, size)
        w_f.result()
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertEqual(out_que1.size, size)
        self.assertEqual(out_que2.size, size)
        for i in range(size):
            self.assertEqual(out_que1.get_dict(),
                             {
                                 'url': 'test_url',
                                 'num1': i,
                                 'num2': i + 1
                             })

        for i in range(size):
            self.assertEqual(out_que2.get_dict(),
                             {
                                 'num1': i
                             })
        self.assertEqual(out_que1.size, 0)
        self.assertEqual(out_que2.size, 0)
        self.assertIsNone(out_que1.get())
        self.assertIsNone(out_que2.get())

    def test_schema_cover(self):
        in_que1 = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        in_que2 = DataQueue([('num1', ColumnType.QUEUE), ('num3', ColumnType.QUEUE)])
        out_que1 = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE), ('num3', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num1', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que1, in_que2], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)

        in_que1.put((1, 1))
        in_que1.put((2, 2))
        in_que1.put((3, 3))

        in_que2.put((4, 4))
        in_que2.put((5, 5))
        in_que2.put((6, 6))
        in_que1.seal()
        in_que2.seal()
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)

        for i in range(3):
            self.assertEqual(out_que1.get_dict(),
                             {
                                 'num1': i + 4,
                                 'num2': i + 1,
                                 'num3': i + 4
                             })

        for i in range(3):
            self.assertEqual(out_que2.get_dict(),
                             {
                                 'num1': i + 4
                             })
        self.assertEqual(out_que1.size, 0)
        self.assertEqual(out_que2.size, 0)
        self.assertIsNone(out_que1.get())
        self.assertIsNone(out_que2.get())

    def test_stopped(self):
        in_que1 = DataQueue([('url', ColumnType.SCALAR), ('num1', ColumnType.QUEUE)])
        in_que2 = DataQueue([('url', ColumnType.SCALAR), ('num2', ColumnType.QUEUE)])
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num1', ColumnType.QUEUE)])

        node = create_node(self.node_repr, self.op_pool, [in_que1, in_que2], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        out_que1.seal()
        def write(size):
            for i in range(size):
                if not in_que1.put(('test_url', i)):
                    return
                time.sleep(0.01)
                if not in_que2.put(('test_url', i + 1)):
                    return
            in_que1.seal()
            in_que2.seal()

        size = 10
        w_f = self.thread_pool.submit(write, size)
        w_f.result()
        f.result()
        self.assertTrue(node.status == NodeStatus.STOPPED)
        self.assertEqual(out_que1.size, 0)
        self.assertEqual(out_que2.size, 0)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)

