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
import copy
from concurrent.futures import ThreadPoolExecutor

from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.nodes import create_node, NodeStatus
from towhee.runtime.data_queue import DataQueue, ColumnType
from towhee.runtime.operator_manager import OperatorPool


class TestFlatMapNode(unittest.TestCase):
    """
    FlatMap node test.
    """
    node_info = {
    'inputs': ('num', ),
    'outputs': ('res', ),
    'op_info': {
        'type': 'hub',
        'operator': 'local/add2',
        'tag': 'main',
        'init_args': [],
        'init_kws': {'factor': 1}
    },
    'iter_info': {
        'type': 'flat_map',
        'param': {}
    },
    'config': {},
    'next_nodes': ['_output']
    }
    node_repr = NodeRepr.from_dict('test_node', node_info)
    op_pool = OperatorPool()
    thread_pool = ThreadPoolExecutor()

    def test_normal(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', [0,1,2]))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('res', ColumnType.QUEUE)])
        out_que2 = DataQueue([('res', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        for i in range(3):
            self.assertEqual(out_que1.get_dict(),
                            {
                                'url': 'test_url',
                                'num': [0,1,2] if i == 0 else None,
                                'res': i + 1
                            })

            self.assertEqual(out_que2.get_dict(),
                            {
                                'res': i + 1
                            })

    def test_multithread(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('res', ColumnType.QUEUE)])
        out_que2 = DataQueue([('res', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)

        def write(size):
            for i in range(size):
                in_que.put(('test_url', range(i, i+3)))
                time.sleep(0.01)
            in_que.seal()

        size = 5
        w_f = self.thread_pool.submit(write, size)
        w_f.result()
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertEqual(out_que1.size, size*3)
        self.assertEqual(out_que2.size, size*3)
        for i in range(size*3):
            self.assertEqual(out_que1.get_dict(),
                             {
                                 'url': 'test_url',
                                 'num': range(i, i+3) if i < 5 else None,
                                 'res': i//3 + i%3 + 1
                             })
        for i in range(size * 3):
            self.assertEqual(out_que2.get_dict(),
                             {
                                 'res': i//3 + i%3 + 1
                             })
        self.assertEqual(out_que1.size, 0)
        self.assertEqual(out_que2.size, 0)
        self.assertIsNone(out_que1.get())
        self.assertIsNone(out_que2.get())

    def test_schema_cover(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', [0,1,2]))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num', ColumnType.QUEUE)])
        node_info = copy.deepcopy(self.node_info)
        node_info['outputs'] = ('num', )
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        for i in range(3):
            self.assertEqual(out_que1.get_dict(),
                            {
                                'url': 'test_url',
                                'num': i + 1,
                            })

            self.assertEqual(out_que2.get_dict(),
                            {
                                'num': i + 1
                            })

    def test_stopped(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', [0, 1, 2]))
        in_que.put(('test_url', [0, 1, 2]))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('res', ColumnType.QUEUE)])
        out_que2 = DataQueue([('res', ColumnType.QUEUE)])
        out_que2.seal()
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.STOPPED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertTrue(out_que1.size == 0)
        self.assertTrue(out_que2.size == 0)

    def test_failed(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', 'a'))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('res', ColumnType.QUEUE)])
        out_que2 = DataQueue([('res', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FAILED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertTrue(out_que1.size == 0)
        self.assertTrue(out_que2.size == 0)

    def test_multi_output(self):
        node_info = {
            'inputs': ('num', ),
            'outputs': ('res1', 'res2'),
            'op_info': {
                'type': 'hub',
                'operator': 'local/multi_flat_map',
                'tag': 'main',
                'init_args': None,
                'init_kws': {'factor': 10}
            },
            'iter_info': {
                'type': 'flat_map',
                'param': None
            },
            'config': {},
            'next_nodes': ['_output']
        }
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', [0, 1, 2]))
        in_que.seal()
        out_que1 = DataQueue([
            ('url', ColumnType.SCALAR),
            ('num', ColumnType.QUEUE),
            ('res1', ColumnType.QUEUE),
            ('res2', ColumnType.QUEUE)
        ])
        out_que2 = DataQueue([('url', ColumnType.SCALAR), ('res2', ColumnType.QUEUE)])
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(node.status, NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        for i in range(3):
            self.assertEqual(out_que1.get_dict(),
                            {
                                'url': 'test_url',
                                'num': [0, 1, 2] if i == 0 else None,
                                'res1': i + 10,
                                'res2': i + 10
                            })
            self.assertEqual(out_que2.get_dict(),
                            {
                                'url': 'test_url',
                                'res2': i + 10
                            })

    def test_multi_output_cover(self):
        node_info = {
            'inputs': ('num', ),
            'outputs': ('num', 'res'),
            'op_info': {
                'type': 'hub',
                'operator': 'local/multi_flat_map',
                'tag': 'main',
                'init_args': None,
                'init_kws': {'factor': 10}
            },
            'iter_info': {
                'type': 'flat_map',
                'param': None
            },
            'config': {},
            'next_nodes': ['_output']
        }
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', [0, 1, 2]))
        in_que.seal()
        out_que1 = DataQueue([
            ('url', ColumnType.SCALAR),
            ('num', ColumnType.QUEUE),
            ('res', ColumnType.QUEUE),
        ])
        out_que2 = DataQueue([('num', ColumnType.QUEUE), ('res', ColumnType.QUEUE)])
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(node.status, NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        for i in range(3):
            self.assertEqual(out_que1.get_dict(),
                            {
                                'url': 'test_url',
                                'num': i + 10,
                                'res': i + 10,
                            })
            self.assertEqual(out_que2.get_dict(),
                            {
                                'num': i + 10,
                                'res': i + 10,
                            })

    def test_generator(self):
        node_info = {
            'inputs': ('num', ),
            'outputs': ('res1', 'res2'),
            'op_info': {
                'type': 'hub',
                'operator': 'local/flat_gen',
                'tag': 'main',
                'init_args': None,
                'init_kws': {'factor': 10}
            },
            'iter_info': {
                'type': 'flat_map',
                'param': None
            },
            'config': {},
            'next_nodes': ['_output']
        }

        in_que = DataQueue([('num', ColumnType.QUEUE)])
        in_que.put(([1, 2], ))
        in_que.seal()
        out_que1 = DataQueue([
            ('num', ColumnType.QUEUE),
            ('res1', ColumnType.QUEUE),
            ('res2', ColumnType.QUEUE)
        ])
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(node.status, NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        for i in [1, 2]:
            self.assertEqual(out_que1.get_dict(),
                            {
                                'num': [1, 2] if i == 1 else None,
                                'res1': i,
                                'res2': i + 10
                            })

    def test_multi_input(self):
        node_info = {
            'inputs': ('num1', 'num2'),
            'outputs': ('res', ),
            'op_info': {
                'type': 'hub',
                'operator': 'local/sum3',
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'flat_map',
                'param': None
            },
            'config': {},
            'next_nodes': ['_output']
        }

        in_que = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        in_que.put(([1, 2], [1, 2]))
        in_que.seal()
        out_que = DataQueue([
            ('num1', ColumnType.QUEUE),
            ('num2', ColumnType.QUEUE),
            ('res', ColumnType.QUEUE),
        ])
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(node.status, NodeStatus.FINISHED)
        self.assertTrue(out_que.sealed)
        for i in range(2):
            self.assertEqual(out_que.get_dict(),
                            {
                                'num1': [1, 2] if i == 0 else None,
                                'num2': [1, 2] if i == 0 else None,
                                'res': (i+1) * 2,
                            })
