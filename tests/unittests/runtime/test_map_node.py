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
import uuid
from concurrent.futures import ThreadPoolExecutor

from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.nodes import create_node, NodeStatus
from towhee.runtime.data_queue import DataQueue, ColumnType, Empty
from towhee.runtime.operator_manager import OperatorPool


class TestMapNode(unittest.TestCase):
    """
    map node test.
    """
    node_info = {
        'inputs': ('num', ),
        'outputs': ('vec', ),
        'op_info': {
            'type': 'hub',
            'operator': 'local/add',
            'tag': 'main',
            'init_args': None,
            'init_kws': {'factor': 10}
        },
        'iter_info': {
            'type': 'map',
            'param': None
        },
        'config': {'name': 'test'},
        'next_nodes': ['_output']
    }

    def setUp(self):
        self.node_repr = NodeRepr.from_dict(uuid.uuid4().hex, self.node_info)
        self.op_pool = OperatorPool()
        self.thread_pool = ThreadPoolExecutor()

    def test_normal(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', 1))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
        out_que2 = DataQueue([('vec', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertEqual(out_que1.get_dict(),
                         {
                             'url': 'test_url',
                             'num': 1,
                             'vec': 11
                         })

        self.assertEqual(out_que2.get_dict(),
                         {
                             'vec': 11
                         })

    def test_multithread(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
        out_que2 = DataQueue([('vec', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)

        def write(size):
            for i in range(size):
                in_que.put(('test_url', i))
                time.sleep(0.01)
            in_que.seal()

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
                                 'num': i,
                                 'vec': 10 + i
                             })

        for i in range(size):
            self.assertEqual(out_que2.get_dict(),
                             {
                                 'vec': 10 + i
                             })
        self.assertEqual(out_que1.size, 0)
        self.assertEqual(out_que2.size, 0)
        self.assertIsNone(out_que1.get())
        self.assertIsNone(out_que2.get())

    def test_schema_cover(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', 1))
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
        self.assertEqual(out_que1.get_dict(),
                         {
                             'url': 'test_url',
                             'num': 11,
                         })

        self.assertEqual(out_que2.get_dict(),
                         {
                             'num': 11
                         })

    def test_stopped(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', 1))
        in_que.put(('test_url', 1))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
        out_que2 = DataQueue([('vec', ColumnType.QUEUE)])
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
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
        out_que2 = DataQueue([('vec', ColumnType.QUEUE)])
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
            'outputs': ('vec1', 'vec2'),
            'op_info': {
                'type': 'hub',
                'operator': 'local/multi_output',
                'tag': 'main',
                'init_args': None,
                'init_kws': {'factor': 10}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', 1))
        in_que.seal()
        out_que1 = DataQueue([
            ('url', ColumnType.SCALAR),
            ('num', ColumnType.QUEUE),
            ('vec1', ColumnType.QUEUE),
            ('vec2', ColumnType.QUEUE)
        ])
        out_que2 = DataQueue([('url', ColumnType.SCALAR), ('vec2', ColumnType.QUEUE)])
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertEqual(out_que1.get_dict(),
                         {
                             'url': 'test_url',
                             'num': 1,
                             'vec1': 11,
                             'vec2': 11
                         })

        self.assertEqual(out_que2.get_dict(),
                         {
                             'url': 'test_url',
                             'vec2': 11
                         })

    def test_multi_output_cover(self):
        node_info = {
            'inputs': ('num', ),
            'outputs': ('num', 'vec'),
            'op_info': {
                'type': 'hub',
                'operator': 'local/multi_output',
                'tag': 'main',
                'init_args': None,
                'init_kws': {'factor': 10}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', 1))
        in_que.seal()
        out_que1 = DataQueue([
            ('url', ColumnType.SCALAR),
            ('num', ColumnType.QUEUE),
            ('vec', ColumnType.QUEUE),
        ])
        out_que2 = DataQueue([('num', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertEqual(out_que1.get_dict(),
                         {
                             'url': 'test_url',
                             'num': 11,
                             'vec': 11
                         })

        self.assertEqual(out_que2.get_dict(),
                         {
                             'num': 11,
                             'vec': 11
                         })

    def test_generator_multi_output(self):
        node_info = {
            'type': 'map',
            'inputs': ('num', ),
            'outputs': ('vec1', 'vec2'),
            'op_info': {
                'type': 'hub',
                'operator': 'local/multi_gen',
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }

        in_que = DataQueue([('num', ColumnType.QUEUE)])
        in_que.put((4, ))
        in_que.seal()
        out_que1 = DataQueue([
            ('num', ColumnType.QUEUE),
            ('vec1', ColumnType.QUEUE),
            ('vec2', ColumnType.QUEUE)
        ])
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        print(node.err_msg)
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertEqual(out_que1.get_dict(),
                         {
                             'num': 4,
                             'vec1': [0, 1, 2, 3],
                             'vec2': [0, 1, 2, 3]
                         })

    def test_generator_single_ouput(self):
        def fun(num):
            i = 0
            while i < num:
                yield i
                i += 1

        node_info = {
            'type': 'map',
            'inputs': ('num', ),
            'outputs': ('nums', ),
            'op_info': {
                'type': 'callable',
                'operator': fun,
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }

        in_que = DataQueue([('num', ColumnType.QUEUE)])
        in_que.put((4, ))
        in_que.seal()
        out_que1 = DataQueue([
            ('num', ColumnType.QUEUE),
            ('nums', ColumnType.QUEUE),
        ])
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        print(node.err_msg)
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertEqual(out_que1.get_dict(),
                         {
                             'num': 4,
                             'nums': [0, 1, 2, 3]
                         })

    def test_all_scalar(self):
        in_que = DataQueue([('num', ColumnType.SCALAR)])
        in_que.put((1, ))
        in_que.seal()
        out_que = DataQueue([('num', ColumnType.SCALAR), ('vec', ColumnType.SCALAR)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que.sealed)
        self.assertEqual(out_que.get_dict(),
                         {
                             'num': 1,
                             'vec': 11,
                         })

    def test_callable(self):
        node_info = copy.deepcopy(self.node_info)
        node_info['op_info']['type'] = 'lambda'
        node_info['op_info']['operator'] = lambda x: x + 1
        node_repr = NodeRepr.from_dict('test_node', node_info)

        in_que = DataQueue([('num', ColumnType.QUEUE)])
        in_que.put((1, ))
        in_que.put((2, ))
        in_que.put((3, ))
        in_que.put((4, ))
        in_que.seal()
        out_que = DataQueue([('vec', ColumnType.QUEUE)])
        node = create_node(node_repr, self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que.sealed)
        for i in range(out_que.size):
            data = out_que.get()
            self.assertEqual(data[0], i + 2)

    def test_create_op_failed(self):
        node_info = copy.deepcopy(self.node_info)
        node_info['op_info']['operator'] = 'mock'
        node_repr = NodeRepr.from_dict('test_node', node_info)
        in_que = DataQueue([('num', ColumnType.QUEUE)])
        node = create_node(node_repr, self.op_pool, [in_que], [])
        self.assertFalse(node.initialize())

        node_info = copy.deepcopy(self.node_info)
        node_info['op_info']['type'] = 'unkown'
        node_repr = NodeRepr.from_dict('test_node', node_info)
        in_que = DataQueue([('num', ColumnType.QUEUE)])
        node = create_node(node_repr, self.op_pool, [in_que], [])
        self.assertFalse(node.initialize())

    def test_no_output(self):
        node_info = {
            'inputs': ('num', ),
            'outputs': (),
            'op_info': {
                'type': 'lambda',
                'operator': print,
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }
        node_repr = NodeRepr.from_dict('test_node', node_info)
        in_que = DataQueue([('num', ColumnType.QUEUE)])
        in_que.put((1, ))
        in_que.put((2, ))
        in_que.seal()
        out_que = DataQueue([])
        node = create_node(node_repr, self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que.sealed)
        self.assertEqual(out_que.size, 0)

    def test_all_empty(self):
        node_info = {
            'inputs': ('num', 'another'),
            'outputs': ('vec', ),
            'op_info': {
                'type': 'lambda',
                'operator': lambda x, y: x + y,
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }
        node_repr = NodeRepr.from_dict('test_node', node_info)
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('another', ColumnType.QUEUE), ('some', ColumnType.QUEUE)])
        in_que.put(('test_url', 1, 1, 1))
        in_que.put(('test_url', Empty(), Empty(), 1))
        in_que.seal()
        out_que = DataQueue(
            [
                ('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('another', ColumnType.QUEUE), ('some', ColumnType.QUEUE),
                ('vec', ColumnType.QUEUE)
            ]
        )
        node = create_node(node_repr, self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(out_que.get_dict(),
                            {
                                'url': 'test_url',
                                'num': 1,
                                'another': 1,
                                'some': 1,
                                'vec': 2,
                            })
        self.assertEqual(out_que.get_dict(),
                            {
                                'url': 'test_url',
                                'num': Empty(),
                                'another': Empty(),
                                'some': 1,
                                'vec': Empty(),
                            })

    def test_scalar_empty(self):
        node_info = {
            'inputs': ('url', 'num', ),
            'outputs': ('vec', ),
            'op_info': {
                'type': 'lambda',
                'operator': lambda x, y : str(x) + str(y),
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }
        node_repr = NodeRepr.from_dict('test_node', node_info)
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('another', ColumnType.QUEUE)])
        in_que.put(('test_url', 1, 1))
        in_que.put(('test_url', Empty(), 1))
        in_que.seal()
        out_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('another', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
        node = create_node(node_repr, self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(out_que.get_dict(),
                            {
                                'url': 'test_url',
                                'num': 1,
                                'another': 1,
                                'vec': 'test_url1',
                            })
        self.assertEqual(out_que.get_dict(),
                            {
                                'url': 'test_url',
                                'num': Empty(),
                                'another': 1,
                                'vec': Empty(),
                            })

    def test_queue_empty(self):
        node_info = {
            'inputs': ('num', 'another'),
            'outputs': ('vec', ),
            'op_info': {
                'type': 'lambda',
                'operator': lambda x, y : x + y,
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }
        node_repr = NodeRepr.from_dict('test_node', node_info)
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('another', ColumnType.QUEUE)])
        in_que.put(('test_url', 1, 1))
        in_que.put(('test_url', Empty(), 1))
        in_que.seal()
        out_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('another', ColumnType.QUEUE), ('vec', ColumnType.QUEUE)])
        node = create_node(node_repr, self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(out_que.get_dict(),
                            {
                                'url': 'test_url',
                                'num': 1,
                                'another': 1,
                                'vec': 2
                            })
        self.assertEqual(out_que.get_dict(),
                            {
                                'url': 'test_url',
                                'num': Empty(),
                                'another': 1,
                                'vec': Empty()
                            })


    def test_empty_without_next(self):
        def fn(x, y):
            return x, y

        node_info = {
            'inputs': ('num', 'another'),
            'outputs': ('res1', 'res2'),
            'op_info': {
                'type': 'lambda',
                'operator': fn,
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'config': {'name': 'test'},
            'next_nodes': None
        }
        node_repr = NodeRepr.from_dict('test_node', node_info)
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('another', ColumnType.QUEUE), ('some', ColumnType.QUEUE)])
        in_que.put(('test_url', 1, 1, 1))
        in_que.put(('test_url', Empty(),1, 1))
        in_que.seal()
        out_que = DataQueue(
            [
                ('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('another', ColumnType.QUEUE), ('some', ColumnType.QUEUE),
                ('res1', ColumnType.QUEUE), ('res2', ColumnType.QUEUE)
            ]
        )
        node = create_node(node_repr, self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(node.status, NodeStatus.FINISHED)
        self.assertEqual(out_que.get_dict(),
                            {
                                'url': 'test_url',
                                'num': 1,
                                'another': 1,
                                'some': 1,
                                'res1': 1,
                                'res2': 1
                            })
        self.assertEqual(out_que.get_dict(),
                            {
                                'url': 'test_url',
                                'num': Empty(),
                                'another': 1,
                                'some': 1,
                                'res1': Empty(),
                                'res2': Empty()
                            })
