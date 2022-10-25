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


class TestFilterNode(unittest.TestCase):
    """
    Filter node test.
    """
    node_info = {
        'inputs': ('num', ),
        'outputs': ('larger_than_5', ),
        'op_info': {
            'type': 'hub',
            'operator': 'local/filter_operator',
            'tag': 'main',
            'init_args': [],
            'init_kws': {'threshold': 5}
        },
        'iter_info': {
            'type': 'filter',
            'param': {'filter_by': ['num']}
        },
        'config': {},
        'next_nodes': ['_output']
    }
    node_repr = NodeRepr.from_dict('test_node', node_info)
    op_pool = OperatorPool()
    thread_pool = ThreadPoolExecutor()

    def test_normal(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        size = 10
        for i in range(size):
            in_que.put(('test_url', i))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('larger_than_5', ColumnType.QUEUE)])
        out_que2 = DataQueue([('larger_than_5', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        node.initialize()
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(node.status, NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        for i in range(size):
            self.assertEqual(out_que1.get_dict(),
                            {
                                'url': 'test_url',
                                'num': i,
                                'larger_than_5': i + 6 if i < 4 else None
                            })
            if i > 5:
                self.assertEqual(out_que2.get_dict(),
                                {
                                    'larger_than_5': i
                                })

    def test_multithread(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('larger_than_5', ColumnType.QUEUE)])
        out_que2 = DataQueue([('larger_than_5', ColumnType.QUEUE)])
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
        self.assertEqual(out_que2.size, size - 6)
        for i in range(size):
            self.assertEqual(out_que1.get_dict(),
                             {
                                 'url': 'test_url',
                                 'num': i,
                                 'larger_than_5': i + 6 if i < 4 else None
                             })
            if i > 5:
                self.assertEqual(out_que2.get_dict(),
                                {
                                    'larger_than_5': i
                                })
        self.assertEqual(out_que1.size, 0)
        self.assertEqual(out_que2.size, 0)
        self.assertIsNone(out_que1.get())
        self.assertIsNone(out_que2.get())

    def test_schema_cover(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        size = 10
        for i in range(size):
            in_que.put(('test_url', i))
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
        self.assertEqual(out_que1.size, 4)
        self.assertEqual(out_que2.size, 4)
        for i in range(6, size):
            self.assertEqual(out_que1.get_dict(),
                            {
                                'url': 'test_url',
                                'num': i
                            })

            self.assertEqual(out_que2.get_dict(),
                            {
                                'num': i
                            })


    def test_schema_cover2(self):
        in_que = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        size = 10
        for i in range(size):
            in_que.put((i, i + 1))
        in_que.seal()
        out_que1 = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num2', ColumnType.QUEUE)])
        node_info = copy.deepcopy(self.node_info)
        node_info['inputs'] = ('num1', 'num2')
        node_info['outputs'] = ('num2', 'num1')
        node_info['iter_info']['param']['filter_by'] = ['num1']
        node_repr = NodeRepr.from_dict('test_node', node_info)
        node = create_node(node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertEqual(out_que1.size, 4)
        self.assertEqual(out_que2.size, 4)
        for i in range(6, size):
            self.assertEqual(out_que1.get_dict(),
                            {
                                'num1': i + 1,
                                'num2': i
                            })

    def test_output_with_scalar(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        size = 10
        for i in range(size):
            in_que.put(('test_url', i))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('larger_than_5', ColumnType.QUEUE)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        for i in range(6, size):
            self.assertEqual(out_que1.get_dict(),
                            {
                                'url': 'test_url',
                                'larger_than_5': i
                            })

    def test_output_all_scalar(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        size = 10
        for i in range(size):
            in_que.put(('test_url', i))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR)])
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que1.size == 1)
        for i in range(size):
            if i == 0:
                self.assertEqual(out_que1.get_dict(),
                                {
                                    'url': 'test_url'
                                })
            else:
                self.assertEqual(out_que1.get_dict(), None)

    def test_stopped(self):
        in_que = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        in_que.put(('test_url', 1))
        in_que.put(('test_url', 1))
        in_que.seal()
        out_que1 = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE), ('larger_than_5', ColumnType.QUEUE)])
        out_que1.seal()
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertEqual(node.status, NodeStatus.STOPPED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que1.size == 0)

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


if __name__ == '__main__':
    unittest.main()
