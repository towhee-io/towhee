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

from towhee.operator.nop import NOPNodeOperator
from towhee.runtime.pipeline import Pipeline
from towhee.runtime.data_queue import Empty
from towhee.runtime.factory import ops, register
from towhee.runtime.runtime_pipeline import RuntimePipeline


# pylint: disable=protected-access
# pylint: disable=unused-variable
# pylint: disable=unnecessary-lambda
@register(name='add_operator')
class AddOperator:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor + x


@register
def sub_operator(x, y):
    return x - y


class TestPipeline(unittest.TestCase):
    """
    Test Pipeline
    """
    def test_property(self):
        pipe1 = Pipeline.input('a', 'b').map('a', 'c', ops.towhee.test_op1('a'))
        self.assertEqual(len(pipe1.dag), 2)

    def test_raise(self):
        with self.assertRaises(ValueError):
            Pipeline.input('a').map('a', 'c', 12).output('c')
        with self.assertRaises(ValueError):
            Pipeline.input({}).map({}, 'c', lambda x: x+1).output('c')
        with self.assertRaises(ValueError):
            Pipeline.input('a').map('a', 'c', lambda x: x+1).output({})
        with self.assertRaises(ValueError):
            Pipeline.input('a,').map('a,', 'c', lambda x: x+1).output('c')
        with self.assertRaises(ValueError):
            Pipeline.input('a,c', 'b').map(('a,c', 'b'), 'c', lambda x, y: x+y).output('c')

    def test_callable_func(self):
        def func(x):
            return x + 2
        pipe = (Pipeline.input('a')
                .map('a', 'c', func)
                .output('c'))
        res = pipe(1).get()
        self.assertEqual(res[0], 3)

    def test_callable_class(self):
        class CallableClass:
            def __init__(self, x):
                self._x = x

            def __call__(self, x):
                return self._x + x

        pipe = (Pipeline.input('a')
                .map('a', 'c', CallableClass(2))
                .output('c'))
        res = pipe(1).get()
        self.assertEqual(res[0], 3)

    def test_lambda(self):
        pipe = (Pipeline.input('a')
                .map('a', 'c', lambda x: x + 2)
                .output('c'))
        res = pipe(2).get()
        self.assertEqual(res[0], 4)

    # TODO: Add pipeline config test
    def test_local(self):
        pipe1 = (Pipeline.input('a')
                 .map('a', 'c', ops.local.add_operator(10))
                 .output('c'))
        c10, = pipe1(2).get()
        self.assertEqual(c10.sum, 12)
        c11, = pipe1(3).get()
        self.assertEqual(c11.sum, 13)

        pipe2 = (Pipeline.input('a')
                 .map('a', 'c', ops.local.add_operator(10))
                 .output('a', 'c'))
        a2, c2 = pipe2(2).get()
        self.assertEqual(a2, 2)
        self.assertEqual(c2.sum, 12)

    def test_registry(self):
        pipe1 = (Pipeline.input('a', 'b')
                 .map('a', 'c', ops.add_operator(10))
                 .map(('b', 'c'), 'd', ops.sub_operator())
                 .output('c', 'd'))
        c1, d1 = pipe1(1, 2).get()
        self.assertEqual(c1, 11)
        self.assertEqual(d1, -9)

    def test_multi_pipe(self):
        pipe1 = Pipeline.input('a', 'b').map('a', 'c', ops.add_operator(10))
        pipe2 = pipe1.map(('b', 'c'), 'd', ops.sub_operator())
        pipe3 = pipe2.output('c', 'd')
        c1, d1 = pipe3(1, 2).get()
        self.assertEqual(c1, 11)
        self.assertEqual(d1, -9)

        self.assertEqual(len(pipe1.dag), 2)
        self.assertEqual(len(pipe2.dag), 3)
        self.assertIsInstance(pipe3, RuntimePipeline)

    def test_concat(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.map('a', 'd', ops.add_operator(10))
        pipe2 = pipe0.map(('b', 'c'), 'e', ops.sub_operator())
        pipe3 = pipe2.concat(pipe1)
        pipe4 = pipe3.output('d', 'e')
        c1, d1 = pipe4(1, 2, 3).get()
        self.assertEqual(c1, 11)
        self.assertEqual(d1, -1)

        self.assertEqual(len(pipe1.dag), 2)
        self.assertEqual(len(pipe2.dag), 2)
        self.assertEqual(len(pipe3.dag), 4)
        self.assertIsInstance(pipe4, RuntimePipeline)

    def test_concat_multi_schema(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.map('a', 'd', ops.add_operator(10))
        pipe2 = pipe0.map(('b', 'c'), 'e', ops.sub_operator())
        pipe3 = pipe2.concat(pipe1).output('a', 'b', 'c', 'd', 'e')
        a, b, c, d, e = pipe3(1, 2, 3).get()
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        self.assertEqual(c, 3)
        self.assertEqual(d, 11)
        self.assertEqual(e, -1)

    def test_concat_updated_schema1(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.map('a', 'a', ops.add_operator(10))
        pipe2 = pipe0.map(('b', 'c'), 'b', ops.sub_operator())
        pipe3 = pipe1.concat(pipe2).output('a', 'b', 'c')
        a, b, c = pipe3(1, 2, 3).get()
        self.assertEqual(a, 1)
        self.assertEqual(b, -1)
        self.assertEqual(c, 3)

        pipe3 = pipe2.concat(pipe1).output('a', 'b', 'c')
        a, b, c = pipe3(1, 2, 3).get()
        self.assertEqual(a, 11)
        self.assertEqual(b, 2)
        self.assertEqual(c, 3)

    def test_concat_updated_schema2(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.map('a', 'a', ops.add_operator(10))
        pipe2 = pipe0.map(('b', 'c'), 'a', ops.sub_operator())
        pipe3 = pipe2.concat(pipe1).output('a')
        a, = pipe3(1, 2, 3).get()
        self.assertEqual(a, 11)

        pipe3 = pipe1.concat(pipe2).output('a')
        a, = pipe3(1, 2, 3).get()
        self.assertEqual(a, -1)

    def test_concat_multi_pipe(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.map('a', 'd', ops.add_operator(10))
        pipe2 = pipe0.map(('b', 'c'), 'e', ops.sub_operator())
        pipe3 = pipe0.map(('b', 'c'), 'f', lambda x, y: x * y)
        pipe4 = pipe3.concat(pipe1, pipe2).output('d', 'e', 'f')

        d, e, f = pipe4(1, 2, 3).get()
        self.assertEqual(d, 11)
        self.assertEqual(e, -1)
        self.assertEqual(f, 6)

    def test_flat_map(self):
        pipe = (Pipeline.input('a')
                .flat_map('a', 'b', ops.local.flat_operator())
                .output('b'))
        res = pipe([1, 2])
        self.assertEqual(res.get()[0].num, 1)
        self.assertEqual(res.get()[0].num, 2)

    def test_filter(self):
        pipe = (Pipeline.input('a')
                .filter('a', 'b', 'a', ops.local.filter_operator(5))
                .output('b'))
        res = pipe(5)
        self.assertEqual(res.get(), None)
        res = pipe(7)
        self.assertEqual(res.get(), [7])

    def test_multi_filter(self):
        def filter_func(x, y):
            return x > 10 and y > 5
        pipe = (Pipeline.input('a', 'b', 'c')
                .map('c', 'c', lambda x: x+1)
                .filter('c', 'd', ('a', 'b'), filter_func)
                .output('a', 'b', 'c', 'd'))
        res = pipe(5, 6, 7)
        self.assertEqual(res.get(), None)
        res = pipe(15, 6, 7)
        self.assertEqual(res.get(), [15, 6, 8, 8])

    def test_time_window(self):
        """
        data: [(0, 1, 0), (1, 2, 1000), (2, 3, 2000), (92, 93, 92000), (93, 94, 93000), (94, 95, 94000), (95, 96, 95000), (96, 97, 96000),
                                                      (97, 98, 97000), (98, 99, 98000), (99, 100, 99000)]
        window_data:
        [(0, 1, 0), (1, 2, 1000), (2, 3, 2000),
        (92, 93, 92000), (93, 94, 93000), (94, 95, 94000), (95, 96, 95000), (96, 97, 96000), (97, 98, 97000), (98, 99, 98000), (99, 100, 99000),
        (95, 96, 95000), (96, 97, 96000), (97, 98, 97000), (98, 99, 98000), (99, 100, 99000)]
        """
        pipe = (Pipeline.input('d')
                .flat_map('d', ('n1', 'n2', 't'), lambda x: ((a, b, c) for a, b, c in x))
                .time_window(('n1', 'n2'), ('s1', 's2'), 't', 10, 5, ops.local.sum2())
                .output('s1', 's2'))
        data = [(i, i+1, i * 1000) for i in range(100) if i < 3 or i > 91]
        res = pipe(data)
        self.assertEqual(res.get(), [3, 6])
        self.assertEqual(res.get(), [764, 772])
        self.assertEqual(res.get(), [485, 490])

    def test_window(self):
        pipe = (Pipeline.input('n1', 'n2')
                .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: list(zip(x, y)))
                .window(('n1', 'n2'), ('s1', 's2'), 2, 1, ops.local.sum2())
                .output('s1', 's2'))
        res = pipe([10, 20, 30, 40], [1, 2, 3, 4])
        self.assertEqual(res.get(), [30, 3])
        self.assertEqual(res.get(), [50, 5])

        pipe = (Pipeline.input('n1', 'n2')
                .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: list(zip(x, y)))
                .window(('n1', 'n2'), ('s1', 's2'), 2, 2, ops.local.sum2())
                .output('s1', 's2'))
        res = pipe([10, 20, 30, 40], [1, 2, 3, 4])
        self.assertEqual(res.get(), [30, 3])
        self.assertEqual(res.get(), [70, 7])

        p = (Pipeline.input('x')
             .map('x', 'z', lambda x: [i+1 for i in range(x)])
             .flat_map('z', 'x', NOPNodeOperator())
             .window('x', 'y', 2, 1, lambda x: sum(x))
             .map('y', 'z', lambda x: x * 10)
             .output('z'))
        res = p(4)
        self.assertEqual(res.to_list(), [[30], [50], [70], [40]])

    def test_window_all(self):
        pipe = (Pipeline.input('n1', 'n2')
                .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: list(zip(x, y)))
                .window_all(('n1', 'n2'), ('s1', 's2'), ops.local.sum2())
                .output('s1', 's2'))
        res = pipe([1, 2, 3, 4], [2, 3, 4, 5])
        self.assertEqual(res.get(), [10, 14])

        pipe = (Pipeline.input('x')
                        .map('x', 'z', lambda x: [i + 1 for i in range(x)])
                        .flat_map('z', 'x', NOPNodeOperator())
                        .window_all('x', 'y', lambda x: sum(x))
                        .map('y', 'z', lambda x: x * 10)
                        .map('z', 'x', NOPNodeOperator())
                        .output('x'))
        res = pipe(4)
        self.assertEqual(res.get(), [100])

    def test_concat_multi_types(self):
        def f(x):
            x = len(x)
            return x, x + 1, x + 2

        p2 = Pipeline.input('p').flat_map('p', 'f', lambda x: ((a, b) for a, b in x))
        p3 = p2.map('f', 't', lambda x: x[1]).time_window('f', 'e', 't', 3, 3, lambda x: len(x))  # pylint: disable=unnecessary-lambda
        p4 = p2.window_all('f', ('l', 's', 'v'), f)

        pipe = p4.concat(p3).output('e', 'l', 's', 'v')
        data = [(i, i * 1000) for i in range(10) if i < 3 or i > 7]
        res = pipe(data)
        self.assertEqual(res.get(), [3, 5, 6, 7])
        self.assertEqual(res.get(), [1, 5, 6, 7])
        self.assertEqual(res.get(), [1, 5, 6, 7])
        self.assertEqual(res.get(), None)

    def test_concat_raise(self):
        p1 = Pipeline.input('a').map('a', 'b', lambda x: x + 1)
        with self.assertRaises(ValueError):
            p1.concat().concat().output('b')
        with self.assertRaises(ValueError):
            p1.concat('a').output('b')

    def test_batch_single_input(self):
        pipe = (Pipeline.input('a')
                .map('a', 'c', lambda x: x + 2)
                .output('c'))
        batch_inputs = [1, 2, 3, 4]
        res = pipe.batch(batch_inputs)
        for idx, item in enumerate(res):
            self.assertEqual(item.get()[0], batch_inputs[idx] + 2)

    def test_batch_multi_inputs(self):
        pipe = (Pipeline.input('a', 'b')
                .flat_map('a', 'c', lambda x: x)
                .map(('c', 'b'), 'd', lambda x, y: x + y)
                .window_all('d', 'd', sum)
                .output('d'))

        batch_inputs = [[[1, 2], 10], [[1, 2, 3], 1], [[1, 3, 5], 2]]
        expect = [23, 9, 15]
        res = pipe.batch(batch_inputs)
        for idx, item in enumerate(res):
            print(item.get()[0], expect[idx])

    def test_batch_with_exception(self):
        p1 = Pipeline.input('a').map('a', 'b', lambda x: x + 1).output()
        with self.assertRaises(RuntimeError) as e:
            p1.batch([1, 'a', 2])

    def test_filter_coverage(self):
        p = (
            Pipeline.input('a')
            .flat_map('a', 'a', lambda x: list(range(1, x + 1)))
            .map('a', 'b', lambda x: x + 1)
            .filter('b', 'a', 'b', lambda x: x > 3)
            .output('a')
        )
        self.assertEqual(p(4).to_list(), [[4], [5]])
        self.assertEqual(p(8).to_list(), [[4], [5], [6], [7], [8], [9]])

    def test_map_coverage(self):
        p = (
            Pipeline.input('a')
            .flat_map('a', 'a', lambda x: list(range(1, x + 1)))
            .window('a', 'b', 3, 3, sum)
            .map('b', 'a', lambda x: x * 10)
            .output('a')
        )
        res = p(4).to_list()
        self.assertEqual(res, [[60], [40]])

    def test_window_all_coverage(self):
        p = (
            Pipeline.input('a')
            .flat_map('a', 'a', lambda x: list(range(1, x + 1)))
            .filter('a', 'b', 'a', lambda x: x > 100)
            .window_all('b', 'a', sum)
            .output('a')
        )
        res = p(4).to_list()
        self.assertEqual(res, [])

    def test_window_coverage(self):
        p = (
            Pipeline.input('a')
            .flat_map('a', 'a', lambda x: list(range(1, x + 1)))
            .window('a', 'b', 3, 3, sum)
            .window('b', 'a', 3, 3, sum)
            .output('a')
        )
        res = p(4).to_list()
        self.assertEqual(res, [[10]])

    def test_time_window_coverage(self):
        """
        The size of timestamp col is less than a2 col.
        """
        p_start = Pipeline.input('a')
        p1 = p_start.flat_map('a', 'a1', lambda x: list(range(1, x + 1)))
        p2 = p_start.flat_map('a', 'a2', lambda x: list(range(1, x + 4)))
        p = (
            p1.concat(p2)
            .map('a1', 'ts', lambda x: x * 1000)
            .time_window('a1', 'b', 'ts', 3, 3, sum)
            .output('b', 'a2')
        )
        res = p(4).to_list()
        self.assertEqual(res, [[3, 1], [7, 2], [Empty(), 3], [Empty(), 4], [Empty(), 5], [Empty(), 6], [Empty(), 7]])

    def test_flatmap_coverage(self):
        p = (
            Pipeline.input('a')
            .flat_map('a', 'a', lambda x: list(range(1, x + 1)))
            .window('a', 'b', 3, 3, sum)
            .flat_map('b', 'a', lambda x: [x * 10])
            .output('a')
        )
        res = p(4).to_list()
        self.assertEqual(res, [[60], [40]])
