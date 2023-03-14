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

# pylint: disable=unnecessary-lambda
p1 = (pipe.input('a')
          .map('a', 'a', lambda x: x+1)
          .map('a', 'a', lambda x: x*10)
          .output('a'))

p2 = (pipe.input('a')
          .map('a', 'b', lambda x: x+1)
          .map(('a', 'b'), 'c', lambda x, y: x+y)
          .output('c'))

p3 = (pipe.input('a', 'b')
          .map(('a', 'b'), 'c', lambda x, y: x+y)
          .output('c'))

p4 = (pipe.input('x')
          .map('x', 'z', lambda x: [i+1 for i in range(x)])
          .output('z'))

p5 = (pipe.input('x')
          .map('x', 'y', lambda x: sum(x))
          .map('y', 'z', lambda x: x*10)
          .output('z'))

p6 = (pipe.input('x')
          .flat_map('x', 'y', p4)
          .map('y', 'z', lambda x: x*100)
          .filter('z', 'z', 'y', lambda x: x > 3)
          .output('z'))

p7 = (pipe.input('x')
          .flat_map('x', 'x', p4)
          .map('x', 'y', lambda x: x*1000)
          .time_window('x', 'x', 'y', 2, 2, p5)
          .output('x'))


class TestDAGPipeline(unittest.TestCase):
    """
    Test RuntimePipeline in Pipeline
    """
    def test_map(self):
        p = (pipe.input('a1')
             .map('a1', 'b1', p1)
             .map(('a1', 'b1'), 'c1', p3)
             .map('c1', 'c1', p2)
             .output('c1')
             )
        res = p(10).get()
        self.assertEqual(res, [241])

    def test_map_dup(self):
        p = (pipe.input('a')  # 10
             .map('a', 'b', p1)  # 110
             .map(('a', 'b'), 'c', p3)  # 120
             .map('c', 'c', p2)  # 121+120=241
             .output('c')
             )
        res = p(10).get()
        self.assertEqual(res, [241])

        p = (pipe.input('a')  # 10
             .map('a', 'a', p1)  # 110
             .map(('a', 'a'), 'a', p3)  # 220
             .map('a', 'a', p2)  # 221+220=441
             .output('a')
             )
        res = p(10).get()
        self.assertEqual(res, [441])

    def test_map_schema(self):
        def add(*inputs):
            re = 0
            for i in inputs:
                re += i
            return re

        p_1 = (pipe.input('x', 'y', 'z')
                   .map(('x', 'y', 'z'), 'z', add)
                   .output('z'))
        p = (pipe.input('x')
                 .map('x', 'y', p_1)
                 .map('y', 'y', lambda x: x*10)
                 .output('y'))
        res = p(1).get()
        self.assertEqual(res, [10])

        p_2 = (pipe.input('a')
               .map('a', 'a', add)
               .map('a', 'a', lambda x: x*10)
               .output('a'))
        p = (pipe.input('x', 'y', 'z')
             .map(('x', 'y', 'z'), 'y', p_2)
             .map('x', 'z', lambda x: x * 10)
             .output('y', 'z'))
        res = p(1, 2, 3).get()
        self.assertEqual(res, [60, 10])

    def test_output(self):
        p_1 = (pipe.input('x', 'y')
                   .map(('x', 'y'), 'z', lambda x, y: x+y)
                   .map('x', 'y', lambda x: x+1)
                   .output('z'))

        p = (pipe.input('x', 'y')
                 .map(('x', 'y'), 'z', p_1)
                 .output('z')
             )
        res = p(1, 2).get()
        self.assertEqual(res, [3])

    def test_multi_output(self):
        p_1 = (pipe.input('x')
                   .map('x', ('x', 'y', 'z'), lambda x: (x, x+2, x+3))
                   .map('x', 'x', lambda x: x+1)
                   .output('x', 'y', 'z'))

        p = (pipe.input('x')
                 .map('x', ('x', 'y', 'z'), p_1)
                 .output('x', 'y', 'z')
             )
        res = p(1).get()
        self.assertEqual(res, [2, 3, 4])

        p = (pipe.input('x')
             .map('x', 'z', p_1)
             .output('z')
             )
        res = p(1).get()
        self.assertEqual(res, [(2, 3, 4)])

    def test_flat_map(self):
        p = (pipe.input('x')
                 .flat_map('x', 'y', p4)
                 .output('y'))
        res = p(3).to_list()
        self.assertEqual(res, [[1], [2], [3]])

    def test_filter(self):
        p_1 = (pipe.input('x')
                   .map('x', 'y', lambda x: x > 3)
                   .output('y'))
        p = (pipe.input('x')
                 .flat_map('x', 'x', p4)
                 .filter('x', 'x', 'x', p_1)
                 .output('x'))
        res = p(4).to_list()
        self.assertEqual(res, [[4]])

        p = (pipe.input('x')
             .flat_map('x', 'y', p4)
             .map('y', 'z', lambda x: x*100)
             .filter('z', 'zr', 'y', p_1)
             .output('zr'))
        res = p(4).to_list()
        self.assertEqual(res, [[400]])

    def test_window(self):
        p = (pipe.input('x')
                 .flat_map('x', 'x', p4)
                 .window('x', 'x', 2, 1, p5)
                 .output('x'))

        res = p(4).to_list()
        self.assertEqual(res, [[30], [50], [70], [40]])

    def test_time_window(self):
        p = (pipe.input('x')
                 .flat_map('x', 'x', p4)
                 .map('x', 'y', lambda x: x*1000)
                 .time_window('x', 'x', 'y', 2, 2, p5)
                 .output('x'))

        res = p(4).to_list()
        self.assertEqual(res, [[10], [50], [40]])

    def test_window_all(self):
        p = (pipe.input('x')
                 .flat_map('x', 'x', p4)
                 .window_all('x', 'x', p5)
                 .output('x'))

        res = p(4).to_list()
        self.assertEqual(res, [[100]])

    def test_nested_pipelines(self):
        p = (pipe.input('y')
                 .map('y', 'f', p6)
                 .output('f')
             )
        res = p(4).to_list()
        self.assertEqual(res, [[400]])

        p = (pipe.input('y')
             .map('y', 'f', p7)
             .output('f')
             )
        res = p(4).to_list()
        self.assertEqual(res, [[10], [50], [40]])

    def test_concat_pipeline(self):
        p_0 = pipe.input('a', 'b', 'c')
        p_1 = p_0.map('a', 'd', lambda x: x + 1)
        p_2 = p_0.map(('b', 'c'), 'e', lambda x, y: x - y)
        p_3 = p_2.concat(p_1).output('d', 'e')

        p = (pipe.input('x', 'y', 'z')
             .map(('x', 'y', 'z'), 'x', p_3)
             .output('x')
             )
        res = p(1, 2, 3).to_list()
        self.assertEqual(res, [[(2, -1)]])

        p_0 = pipe.input('a', 'b', 'c')
        p_1 = p_0.map('a', 'd', p2)
        p_2 = p_0.map(('b', 'c'), 'e', p3)
        p_3 = p_2.concat(p_1).output('d', 'e')

        res = p_3(1, 2, 3).to_list()
        self.assertEqual(res, [[3, 5]])
