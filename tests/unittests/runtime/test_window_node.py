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

from towhee import register
from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.nodes import create_node, NodeStatus
from towhee.runtime.nodes._window import _WindowBuffer, _WindowReader
from towhee.runtime.data_queue import DataQueue, ColumnType
from towhee.engine.operator_pool import OperatorPool


class TestWindowBuffer(unittest.TestCase):
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


class TestReader(unittest.TestCase):
    def test_window_reader(self):
        q = DataQueue([('url', ColumnType.SCALAR), ('num', ColumnType.QUEUE)])
        for i in range(20):
            q.put(('test', i))
        q.seal()

        reader = _WindowReader(2, 10, q)
        outputs = [{'url': ['test', 'test'], 'num': [0, 1]}, {'url': ['test', 'test'], 'num': [10, 11]}]
        index = 0
        while True:
            data = reader.read()
            if data is None:
                break
            self.assertEqual(outputs[index], data)
            index += 1


class TestWindowNode(unittest.TestCase):
    '''
    map node test.
    '''
    node_info = {
        'name': 'test_node',
        'inputs': ('num1', 'num2'),
        'outputs': ('sum1', 'sum2'),
        'op_info': {
            'type': 'local',
            'operator': 'local/sum2',
            'tag': 'main',
            'init_args': None,
            'init_kws': {}
        },
        'iter_info': {
            'type': 'window',
            'param': {
                'size': 4,
                'step': 3
            }
        },
        'config': {},
        'next_nodes': ['_output']
    }
    node_repr = NodeRepr.from_dict(node_info)
    op_pool = OperatorPool()
    thread_pool = ThreadPoolExecutor()

    def test_normal(self):
        in_que = DataQueue([('num1', ColumnType.SCALAR), ('num2', ColumnType.QUEUE)])
        in_que.put((1, 1))
        in_que.put((1, 2))
        in_que.put((1, 3))
        in_que.seal()
        out_que1 = DataQueue([('num1', ColumnType.SCALAR),
                              ('num2', ColumnType.QUEUE),
                              ('sum1', ColumnType.QUEUE),
                              ('sum2', ColumnType.QUEUE)])
        out_que2 = DataQueue([('num1', ColumnType.SCALAR),
                              ('sum1', ColumnType.QUEUE),
                              ('sum2', ColumnType.QUEUE)])
        
        node = create_node(self.node_repr, self.op_pool, [in_que], [out_que1, out_que2])
        self.assertTrue(node.initialize())
        f = self.thread_pool.submit(node.process)
        f.result()
        self.assertTrue(node.status == NodeStatus.FINISHED)
        self.assertTrue(out_que1.sealed)
        self.assertTrue(out_que2.sealed)
        while True:
            if out_que1.size == 0:
                break
            print(out_que1.get_dict())

        while True:
            if out_que2.size == 0:
                break
            print(out_que2.get_dict())            
            
        # print(out_que2.get_dict())
        # self.assertEqual(out_que1.get_dict(),
        #                  {
        #                      'url': 'test_url',
        #                      'num': 1,
        #                      'vec': 11
        #                  })

        # self.assertEqual(out_que2.get_dict(),
        #                  {
        #                      'vec': 11
        #                  })    
