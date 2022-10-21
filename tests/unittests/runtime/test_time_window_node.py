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
import copy
from concurrent.futures import ThreadPoolExecutor

from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.nodes import create_node, NodeStatus
from towhee.runtime.nodes._time_window import _TimeWindowBuffer
from towhee.runtime.data_queue import DataQueue, ColumnType
from towhee.engine.operator_pool import OperatorPool


class TestTimeWindowBuffer(unittest.TestCase):
    """TimeWindowBufer
    """

    def _in_test_function(self, time_range, time_step, size):
        buf = _TimeWindowBuffer(time_range, time_step)
        ret = [list(range(i, min(i + time_range, size))) for i in range(0, size, time_step)]
        data = [(i, i * 1000) for i in range(size)]
        index = 0
        for item in data:
            if buf(item[0], item[1]):
                self.assertEqual(buf.data, ret[index])
                buf = buf.next()
                index += 1
        while buf and buf.data:
            self.assertEqual(buf.data, ret[index])
            buf = buf.next()
            index += 1

    def test_size_equal_step(self):
        """
        0s -----> 99s
        time_range_sec: 9s
        time_step_sec: 9s
        """
        for i in range(2, 110):
            self._in_test_function(i, i, 100)

    def test_size_greater_than_step(self):
        """
        0s -----> 99s
        time_range_sec: 20s
        time_step_sec: 10s
        """
        for i in range(2, 110):
            self._in_test_function(i + i, i, 100)

    def test_size_less_than_step(self):
        """
        0s -----> 99s
        time_range_sec: 10s
        time_step_sec: 20s
        """
        for i in range(2, 110):
            self._in_test_function(i, i + 1, 100)

    def test_discontinuous_timestamp(self):
        """
        0s --->4s     81s ---> 99s
        time_range_sec: 10s
        time_step_sec: 5s
        """
        buf = _TimeWindowBuffer(10, 5)
        data = [(i, i * 1000) for i in range(100) if i < 3 or i > 91]

        ret = [
            [0, 1, 2],
            [92, 93, 94, 95, 96, 97, 98, 99],
            [95, 96, 97, 98, 99]
        ]
        index = 0
        for item in data:
            if buf(item[0], item[1]):
                self.assertEqual(buf.data, ret[index])
                buf = buf.next()
                index += 1
        while buf and buf.data:
            self.assertEqual(buf.data, ret[index])
            buf = buf.next()
            index += 1


class TestWindowNode(unittest.TestCase):
    '''
    Window
    '''
    node_info = {
        'name': 'test_node',
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
            'type': 'time_window',
            'param': {
                'time_range_sec': 3,
                'time_step_sec': 3,
                'timestamp_col': 'timestamp'
            }
        },
        'config': {},
        'next_nodes': ['_output']
    }
    node_repr = NodeRepr.from_dict('test_node', node_info)
    op_pool = OperatorPool()
    thread_pool = ThreadPoolExecutor()

    def _test_function(self, node_info, exp_ret1, exp_ret2):
        in_que = DataQueue([
            ('num1', ColumnType.SCALAR),
            ('num2', ColumnType.QUEUE),
            ('timestamp', ColumnType.QUEUE)])

        in_que.put((1, 1, 1000))
        in_que.put((1, 2, 2000))
        in_que.put((1, 3, 3000))
        in_que.put((1, 4, 4000))
        in_que.seal()
        out_que1 = DataQueue([('num1', ColumnType.SCALAR),
                              ('num2', ColumnType.QUEUE),
                              ('sum1', ColumnType.QUEUE),
                              ('sum2', ColumnType.QUEUE)])

        out_que2 = DataQueue([('num1', ColumnType.SCALAR),
                              ('sum1', ColumnType.QUEUE),
                              ('sum2', ColumnType.QUEUE)])

        node = create_node(NodeRepr.from_dict('test_node', node_info), self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)

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

        num1: scalar       ----------1------------>
        num2: queue        ----1----2----3---4----->
        timestamp: queue   --1000-2000-3000-4000--->

        params:
            time_range_sec: 3
            time_step_sec: 3

        time_window: [0s, 3s)  [3s, 6s)
        outputs:

        que1:
        num1: scalar   -----1----->
        num2: queue    --1-2-3-4--->
        sum1: queue    --2-2------>
        sum2: queue    --3-7------>

        que2:
        num1: scalar   ------1------>
        sum1: queue    --2-2------>
        sum2: queue    --3-7------>
        """
        exp_ret1 = [
            {'num1': 1, 'num2': 1, 'sum1': 2, 'sum2': 3},
            {'num1': 1, 'num2': 2, 'sum1': 2, 'sum2': 7},
            {'num1': 1, 'num2': 3, 'sum1': None, 'sum2': None},
            {'num1': 1, 'num2': 4, 'sum1': None, 'sum2': None}
        ]

        exp_ret2 = [
            {'num1': 1, 'sum1': 2, 'sum2': 3},
            {'num1': 1, 'sum1': 2, 'sum2': 7}
        ]

        self._test_function(self.node_info, exp_ret1, exp_ret2)

    def test_small(self):
        """
        inputs:

        num1: scalar       ----------1------------>
        num2: queue        ----1----2----3---4----->
        timestamp: queue   --1000-2000-3000-4000--->

        params:
            time_range_sec: 3
            time_step_sec: 2

        time_window: [0s, 3s)  [2s, 5s) [4s, 6s)

        outputs:

        que1:
        num1: scalar   -----1----->
        num2: queue    --1-2-3-4--->
        sum1: queue    --2-3-1----->
        sum2: queue    --3-9-4----->

        que2:
        num1: scalar   ------1------>
        sum1: queue    --2-3-1----->
        sum2: queue    --3-9-4----->
        """

        info = copy.deepcopy(self.node_info)
        info['iter_info']['param']['time_step_sec'] = 2
        exp_ret1 = [
            {'num1': 1, 'num2': 1, 'sum1': 2, 'sum2': 3},
            {'num1': 1, 'num2': 2, 'sum1': 3, 'sum2': 9},
            {'num1': 1, 'num2': 3, 'sum1': 1, 'sum2': 4},
            {'num1': 1, 'num2': 4, 'sum1': None, 'sum2': None}
        ]

        exp_ret2 = [
            {'num1': 1, 'sum1': 2, 'sum2': 3},
            {'num1': 1, 'sum1': 3, 'sum2': 9},
            {'num1': 1, 'sum1': 1, 'sum2': 4}
        ]
        self._test_function(info, exp_ret1, exp_ret2)

    def test_large(self):
        """
        inputs:

        num1: scalar       ----------1------------>
        num2: queue        ----1----2----3---4----->
        timestamp: queue   --1000-2000-3000-4000--->

        params:
            time_range_sec: 2
            time_step_sec: 3

        time_window: [0s, 2s)  [3s, 5s)

        outputs:

        que1:
        num1: scalar   -----1----->
        num2: queue    --1-2-3-4--->
        sum1: queue    --1-2------>
        sum2: queue    --1-7------>

        que2:
        num1: scalar   ------1------>
        sum1: queue    --1-2------>
        sum2: queue    --1-7------>
        """
        exp_ret1 = [
            {'num1': 1, 'num2': 1, 'sum1': 1, 'sum2': 1},
            {'num1': 1, 'num2': 2, 'sum1': 2, 'sum2': 7},
            {'num1': 1, 'num2': 3, 'sum1': None, 'sum2': None},
            {'num1': 1, 'num2': 4, 'sum1': None, 'sum2': None}
        ]

        exp_ret2 = [
            {'num1': 1, 'sum1': 1, 'sum2': 1},
            {'num1': 1, 'sum1': 2, 'sum2': 7}
        ]
        info = copy.deepcopy(self.node_info)
        info['iter_info']['param']['time_range_sec'] = 2
        self._test_function(info, exp_ret1, exp_ret2)
