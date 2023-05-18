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
from towhee.runtime.nodes._window import _WindowBuffer
from towhee.runtime.data_queue import DataQueue, ColumnType, Empty
from towhee.runtime.operator_manager import OperatorPool


class TestWindowBuffer(unittest.TestCase):
    """
    Test for _WindowBuffer
    """
    def test_count_window(self):
        """
        size == step

        Inputs:

          ----1-2-3-4-5-6-7-8-9--->

        size: 2
        step: 2

        outputs:
          ----[1, 2] - [3, 4] - [5, 6] - [7, 8] - [9] ---->
        """
        ret = [[1, 2], [3, 4], [5, 6], [7, 8], [9]]
        index = 0
        buf = _WindowBuffer(2, 2)
        for i in range(1, 10):
            if buf(i, i - 1):
                self.assertEqual(buf.data, ret[index])
                index += 1
                buf = buf.next()

        while buf is not None:
            if buf.data:
                self.assertEqual(buf.data, ret[index])
                index += 1
            buf = buf.next()

    def test_sliding_small(self):
        """
        size > step

        Inputs:

          ----1-2-3-4-5-6-7-8-9--->

        size: 3
        step: 2

        outputs:
          ----[1, 2, 3] - [3, 4, 5] - [5, 6, 7] - [7, 8, 9] - [9] ---->
        """
        ret = [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9]]
        index = 0
        buf = _WindowBuffer(3, 2)
        for i in range(1, 10):
            if buf(i, i - 1):
                self.assertEqual(buf.data, ret[index])
                index += 1
                buf = buf.next()

        while buf is not None:
            if buf.data:
                self.assertEqual(buf.data, ret[index])
                index += 1
            buf = buf.next()


    def test_sliding_large(self):
        """
        size > step

        Inputs:

          ----1-2-3-4-5-6-7-8-9--->

        size: 2
        step: 5

        outputs:
          ----[1, 2] - [6, 7] ----->
        """
        ret = [[1, 2], [6, 7]]
        index = 0
        buf = _WindowBuffer(2, 5)
        for i in range(1, 10):
            if buf(i, i - 1):
                self.assertEqual(buf.data, ret[index])
                index += 1
                buf = buf.next()

        while buf is not None:
            if buf.data:
                self.assertEqual(buf.data, ret[index])
                index += 1
            buf = buf.next()

class TestWindowNode(unittest.TestCase):
    '''
    Window
    '''
    node_info = {
        'inputs': ('num1', 'num2'),
        'outputs': ('sum1', 'sum2'),
        'op_info': {
            'type': 'hub',
            'operator': 'local/sum2',
            'tag': 'main',
            'init_args': None,
            'init_kws': {}
        },
        'iter_info': {
            'type': 'window',
            'param': {
                'size': 3,
                'step': 3
            }
        },
        'config': {'name': 'test'},
        'next_nodes': ['_output']
    }

    def setUp(self):
        self.node_repr = NodeRepr(uid=uuid.uuid4().hex, **self.node_info)
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
                              ('sum1', ColumnType.QUEUE),
                              ('sum2', ColumnType.QUEUE)])

        out_que2 = DataQueue([('num1', ColumnType.SCALAR),
                              ('sum1', ColumnType.QUEUE),
                              ('sum2', ColumnType.QUEUE)])

        node = create_node(NodeRepr(uid='test_node', **node_info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertTrue(out_que1.size == 4)
        self.assertTrue(out_que2.size == 2)

        ret = []
        while out_que1.size > 0:
            ret.append(out_que1.get_dict())
        self.assertEqual(ret, exp_ret1)

        ret = []
        while out_que2.size > 0:
            ret.append(out_que2.get_dict())
        self.assertEqual(ret, exp_ret2)

    def test_equal(self):
        """
        inputs:

        num1: scalar  -----1----->
        num2: queue   --1-2-3-4--->

        params:
            size: 3
            step: 3

        outputs:

        que1:
        num1: scalar   -----1----->
        num2: queue    --1-2-3-4--->
        sum1: queue    --3-1------>
        sum2: queue    --6-4------>

        que2:
        num1: scalar   ------1------>
        sum1: queue    --3-1------>
        sum2: queue    --6-4------>
        """
        exp_ret1 = [
            {'num1': 1, 'num2': 1, 'sum1': 3, 'sum2': 6},
            {'num1': 1, 'num2': 2, 'sum1': 1, 'sum2': 4},
            {'num1': 1, 'num2': 3, 'sum1': Empty(), 'sum2': Empty()},
            {'num1': 1, 'num2': 4, 'sum1': Empty(), 'sum2': Empty()}
        ]

        exp_ret2 = [
            {'num1': 1, 'sum1': 3, 'sum2': 6},
            {'num1': 1, 'sum1': 1, 'sum2': 4}
        ]

        self._test_function(self.node_info, exp_ret1, exp_ret2)

    def test_small(self):
        """
        inputs:

        num1: scalar  -----1----->
        num2: queue   --1-2-3-4--->

        params:
            size: 3
            step: 2

        outputs:

        que1:
        num1: scalar   -----1----->
        num2: queue    --1-2-3-4--->
        sum1: queue    --3-2------>
        sum2: queue    --6-7------>

        que2:
        num1: scalar   ------1------>
        sum1: queue    --3-2------>
        sum2: queue    --6-7------>
        """

        info = copy.deepcopy(self.node_info)
        info['iter_info']['param']['step'] = 2
        exp_ret1 = [
            {'num1': 1, 'num2': 1, 'sum1': 3, 'sum2': 6},
            {'num1': 1, 'num2': 2, 'sum1': 2, 'sum2': 7},
            {'num1': 1, 'num2': 3, 'sum1': Empty(), 'sum2': Empty()},
            {'num1': 1, 'num2': 4, 'sum1': Empty(), 'sum2': Empty()}
        ]

        exp_ret2 = [
            {'num1': 1, 'sum1': 3, 'sum2': 6},
            {'num1': 1, 'sum1': 2, 'sum2': 7}
        ]
        self._test_function(info, exp_ret1, exp_ret2)

    def test_large(self):
        """
        inputs:

        num1: scalar  -----1----->
        num2: queue   --1-2-3-4--->

        params:
            size: 2
            step: 3

        outputs:

        que1:
        num1: scalar   -----1----->
        num2: queue    --1-2-3-4--->
        sum1: queue    --2-1------>
        sum2: queue    --3-4------>

        que2:
        num1: scalar   ------1------>
        sum1: queue    --2-1------>
        sum2: queue    --3-4------>
        """
        exp_ret1 = [
            {'num1': 1, 'num2': 1, 'sum1': 2, 'sum2': 3},
            {'num1': 1, 'num2': 2, 'sum1': 1, 'sum2': 4},
            {'num1': 1, 'num2': 3, 'sum1': Empty(), 'sum2': Empty()},
            {'num1': 1, 'num2': 4, 'sum1': Empty(), 'sum2': Empty()}
        ]

        exp_ret2 = [
            {'num1': 1, 'sum1': 2, 'sum2': 3},
            {'num1': 1, 'sum1': 1, 'sum2': 4}
        ]
        info = copy.deepcopy(self.node_info)
        info['iter_info']['param']['size'] = 2
        self._test_function(info, exp_ret1, exp_ret2)

    def test_multithread(self):
        in_que = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])

        out_que1 = DataQueue([('num1', ColumnType.QUEUE),
                              ('num2', ColumnType.QUEUE),
                              ('sum1', ColumnType.QUEUE),
                              ('sum2', ColumnType.QUEUE)])

        out_que2 = DataQueue([('sum2', ColumnType.QUEUE)])

        info = copy.deepcopy(self.node_info)
        info['iter_info']['param']['size'] = 2
        info['iter_info']['param']['step'] = 5
        node = create_node(NodeRepr(uid='test_node', **info), self.op_pool, [in_que], [out_que1, out_que2])
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
            {'num1': 1, 'num2': 0, 'sum1': 3, 'sum2': 1},
            {'num1': 2, 'num2': 1, 'sum1': Empty(), 'sum2': Empty()},
            {'num1': 3, 'num2': 2, 'sum1': Empty(), 'sum2': Empty()},
            {'num1': 4, 'num2': 3, 'sum1': Empty(), 'sum2': Empty()},
            {'num1': 5, 'num2': 4, 'sum1': Empty(), 'sum2': Empty()}
        ]

        exp_ret2 = [
            {'sum2': 1}
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
        out_que1 = DataQueue([('num1', ColumnType.QUEUE),
                              ('num2', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num1', ColumnType.QUEUE)])
        info = copy.deepcopy(self.node_info)
        info['iter_info']['param']['size'] = 2
        info['iter_info']['param']['step'] = 2
        info['outputs'] = ('num1', 'num2')
        node = create_node(NodeRepr(uid='test_node', **info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()

        exp_ret1 = [
            {'num1': 2, 'num2': 3},
            {'num1': 1, 'num2': 3},
        ]

        exp_ret2 = [
            {'num1': 2},
            {'num1': 1}
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

        out_que1 = DataQueue([('num1', ColumnType.QUEUE),
                              ('num2', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num1', ColumnType.QUEUE)])
        info = copy.deepcopy(self.node_info)
        info['iter_info']['param']['size'] = 2
        info['iter_info']['param']['step'] = 2
        info['outputs'] = ('num1', 'num2')
        node = create_node(NodeRepr(uid='test_node', **info), self.op_pool, [in_que], [out_que1, out_que2])
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
        in_que = DataQueue([('num1', ColumnType.SCALAR), ('num2', ColumnType.QUEUE)])
        in_que.put(('a', 1))
        in_que.seal()
        out_que1 = DataQueue([('num1', ColumnType.QUEUE),
                              ('num2', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num1', ColumnType.QUEUE)])
        info = copy.deepcopy(self.node_info)
        info['outputs'] = ('num1', 'num2')
        node = create_node(NodeRepr(uid='test_node', **info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FAILED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        self.assertTrue(out_que1.size == 0)
        self.assertTrue(out_que2.size == 0)

    def test_diff_size(self):
        in_que = DataQueue([('num1', ColumnType.QUEUE), ('num2', ColumnType.QUEUE)])
        in_que.put((1, 2))
        in_que.put((1, Empty()))
        in_que.put((1, 3))
        in_que.put((3, 4))
        in_que.put((5, Empty()))
        in_que.seal()

        out_que = DataQueue([
            ('sum1', ColumnType.QUEUE),
            ('sum2', ColumnType.QUEUE)
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
                'type': 'window',
                'param': {
                    'size': 2,
                    'step': 2
                }
            },
        'config': {'name': 'test'},
            'next_nodes': ['_output']
        }

        node = create_node(NodeRepr(uid='test_node', **node_info), self.op_pool, [in_que], [out_que])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que.sealed)
        self.assertTrue(out_que.size == 3)
        sum1, sum2 = out_que.get()
        self.assertEqual(sum1, 2)
        self.assertEqual(sum2, 5)
        sum1, sum2 = out_que.get()
        self.assertEqual(sum1, 4)
        self.assertEqual(sum2, 4)

        sum1, sum2 = out_que.get()
        self.assertEqual(sum1, 5)
        self.assertEqual(sum2, 0)
