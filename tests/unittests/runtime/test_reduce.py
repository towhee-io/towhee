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
from functools import reduce
from concurrent.futures import ThreadPoolExecutor

from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.nodes import create_node, NodeStatus
from towhee.runtime.data_queue import DataQueue, ColumnType, Empty
from towhee.runtime.operator_manager import OperatorPool


class TestReduce(unittest.TestCase):
    '''
    Reduce
    '''
    node_info = {
        'inputs': ('num1', 'num2'),
        'outputs': ('sum1', 'sum2'),
        'op_info': {
            'type': 'hub',
            'operator': 'local/reduce_op',
            'tag': 'main',
            'init_args': None,
            'init_kws': {}
        },
        'iter_info': {
            'type': 'reduce',
            'param': {
            }
        },
        'config': {'name': 'test'},
        'next_nodes': ['_output']
    }

    def setUp(self):
        self.node_repr = NodeRepr.from_dict(uuid.uuid4().hex, self.node_info)
        self.op_pool = OperatorPool()
        self.thread_pool = ThreadPoolExecutor()

    def _test_function(self, node_info, exp_ret1, exp_ret2):
        in_que = DataQueue([('num1', ColumnType.SCALAR), ('num2', ColumnType.QUEUE)])
        in_que.put((1, 1))
        in_que.put((1, 2))
        in_que.put((1, 3))
        in_que.put((1, 4))
        in_que.seal()
        out_que1 = DataQueue([('num1', ColumnType.SCALAR),
                              ('num2', ColumnType.QUEUE),
                              ('sum1', ColumnType.SCALAR),
                              ('sum2', ColumnType.SCALAR)])

        out_que2 = DataQueue([('num1', ColumnType.SCALAR),
                              ('sum1', ColumnType.SCALAR),
                              ('sum2', ColumnType.SCALAR)])

        node = create_node(NodeRepr.from_dict('test_node', node_info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertTrue(out_que1.size == 4)
        self.assertTrue(out_que2.size == 1)

        ret = []
        while out_que1.size > 0:
            ret.append(out_que1.get_dict())
        self.assertEqual(ret, exp_ret1)

        ret = []
        while out_que2.size > 0:
            ret.append(out_que2.get_dict())
        self.assertEqual(ret, exp_ret2)

    def test_normal(self):
        exp1 = [
            {'num1': 1, 'num2': 1, 'sum1': 4, 'sum2': 10},
            {'num1': 1, 'num2': 2, 'sum1': 4, 'sum2': 10},
            {'num1': 1, 'num2': 3, 'sum1': 4, 'sum2': 10},
            {'num1': 1, 'num2': 4, 'sum1': 4, 'sum2': 10}
        ]
        exp2 = [
            {'num1': 1, 'sum1': 4, 'sum2': 10}
        ]
        self._test_function(self.node_info, exp1, exp2)

    def test_multithread(self):
        in_que = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])

        out_que1 = DataQueue([('num1', ColumnType.QUEUE),
                              ('num2', ColumnType.QUEUE),
                              ('sum1', ColumnType.SCALAR),
                              ('sum2', ColumnType.SCALAR)])

        out_que2 = DataQueue([('sum2', ColumnType.SCALAR)])

        node = create_node(NodeRepr.from_dict('test_node', self.node_info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)

        def write(size):
            for i in range(size):
                in_que.put((i + 1, i))
                time.sleep(0.01)
            in_que.seal()
        size = 5
        w_f = self.thread_pool.submit(write, size)
        w_f.result()
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)

        exp_ret1 = [
            {'num1': 1, 'num2': 0, 'sum1': 15, 'sum2': 10},
            {'num1': 2, 'num2': 1, 'sum1': 15, 'sum2': 10},
            {'num1': 3, 'num2': 2, 'sum1': 15, 'sum2': 10},
            {'num1': 4, 'num2': 3, 'sum1': 15, 'sum2': 10},
            {'num1': 5, 'num2': 4, 'sum1': 15, 'sum2': 10}
        ]

        exp_ret2 = [
            {'sum2': 10}
        ]

        ret1 = []
        while out_que1.size > 0:
            ret1.append(out_que1.get_dict())
        self.assertEqual(ret1, exp_ret1)

        ret2 = []
        while out_que2.size > 0:
            ret2.append(out_que2.get_dict())
        self.assertEqual(ret2, exp_ret2)

    def test_schema_cover(self):
        in_que = DataQueue([('num1', ColumnType.SCALAR), ('num2', ColumnType.QUEUE)])
        in_que.put((1, 1))
        in_que.put((1, 2))
        in_que.put((1, 3))
        in_que.seal()
        out_que1 = DataQueue([('num1', ColumnType.SCALAR),
                              ('num2', ColumnType.SCALAR)])

        out_que2 = DataQueue([('num1', ColumnType.SCALAR)])
        info = copy.deepcopy(self.node_info)

        info['outputs'] = ('num1', 'num2')
        node = create_node(NodeRepr.from_dict('test_node', info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()

        exp_ret1 = [
            {'num1': 3, 'num2': 6}
        ]

        exp_ret2 = [
            {'num1': 3}
        ]

        ret1 = []
        while out_que1.size > 0:
            ret1.append(out_que1.get_dict())
        self.assertEqual(ret1, exp_ret1)

        ret2 = []
        while out_que2.size > 0:
            ret2.append(out_que2.get_dict())
        self.assertEqual(ret2, exp_ret2)

    def test_stopped(self):
        in_que = DataQueue([('num1', ColumnType.SCALAR), ('num2', ColumnType.QUEUE)])
        in_que.put((1, 1))

        out_que1 = DataQueue([('num1', ColumnType.SCALAR),
                              ('num2', ColumnType.SCALAR)])
        out_que2 = DataQueue([('num1', ColumnType.SCALAR)])
        info = copy.deepcopy(self.node_info)
        info['outputs'] = ('num1', 'num2')
        node = create_node(NodeRepr.from_dict('test_node', info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        in_que.put((1, 2))
        in_que.put((1, 3))
        in_que.put((1, 1))
        out_que2.seal()
        in_que.seal()
        f.result()
        self.assertTrue(node.status == NodeStatus.STOPPED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertTrue(out_que1.size == 0)
        self.assertTrue(out_que2.size == 0)

    def test_failed(self):
        in_que = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        in_que.put(('a', 1))
        in_que.put((3, 1))
        in_que.seal()
        out_que1 = DataQueue([('num1', ColumnType.SCALAR),
                              ('num2', ColumnType.SCALAR)])
        out_que2 = DataQueue([('num1', ColumnType.SCALAR)])
        info = copy.deepcopy(self.node_info)
        info['outputs'] = ('num1', 'num2')
        node = create_node(NodeRepr.from_dict('test_node', info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FAILED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertTrue(out_que1.size == 0)
        self.assertTrue(out_que2.size == 0)

    def test_diff_size(self):
        in_que = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE), ('side_by', ColumnType.QUEUE)])
        in_que.put((1, 2, 1))
        in_que.put((1, Empty(), 2))
        in_que.put((1, 3, 3))
        in_que.seal()

        out_que = DataQueue([
            ('sum1', ColumnType.SCALAR),
            ('sum2', ColumnType.SCALAR),
            ('num1', ColumnType.QUEUE),
            ('num2', ColumnType.QUEUE)
        ])

        node_info = {
            'inputs': ('num1', 'num2'),
            'outputs': ('sum1', 'sum2'),
            'op_info': {
                'type': 'lambda',
                'operator': lambda x, y: (sum(x), sum(y)),
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'reduce',
                'param': {
                }
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }

        node = create_node(NodeRepr.from_dict('test_node', node_info), self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que.sealed)
        self.assertTrue(out_que.size == 3)
        sum1, sum2, _, _ = out_que.get()
        self.assertEqual(sum1, 3)
        self.assertEqual(sum2, 5)

    def test_multithread_get(self):
        def reduce_op(*seqs):
            def func(seq):
                ret = 0
                pre_i = -1
                for i in seq:
                    assert pre_i < i
                    pre_i = i
                    ret += i
                return ret

            pool = ThreadPoolExecutor()
            fs = []
            ret = []
            for seq in seqs:
                f = pool.submit(func, seq)
                fs.append(f)
            for f in fs:
                ret.append(f.result())
            return ret[:2]

        in_que = DataQueue([
            ('num1', ColumnType.QUEUE),
            ('num2', ColumnType.QUEUE),
            ('num3', ColumnType.QUEUE),
            ('num4', ColumnType.QUEUE),
            ('side_by', ColumnType.QUEUE)
        ], max_size=-1)

        out_que = DataQueue([
            ('sum1', ColumnType.SCALAR),
            ('sum2', ColumnType.SCALAR),
            ('side_by', ColumnType.QUEUE)
        ])

        node_info = {
            'inputs': ('num1', 'num2'),
            'outputs': ('sum1', 'sum2'),
            'op_info': {
                'type': 'lambda',
                'operator': reduce_op,
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'reduce',
                'param': {
                }
            },
            'config': {'name': 'test'},
            'next_nodes': ['_output']
        }

        num_size = 10000
        node = create_node(NodeRepr.from_dict('test_node', node_info), self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        for i in range(num_size):
            in_que.put([i] * 5)
        in_que.seal()
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que.sealed)
        ret = out_que.to_list()
        self.assertEqual(len(ret), num_size)
        side_by = [item[2] for item in ret]
        def check(x, y):
            assert x < y
            return 0
        reduce(check, side_by)
        self.assertEqual(ret[0][0], 49995000)
        self.assertEqual(ret[0][1], 49995000)
