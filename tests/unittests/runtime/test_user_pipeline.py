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

from towhee.runtime.pipeline import Pipeline
from towhee.runtime.factory import ops, register
from towhee.runtime.runtime_pipeline import RuntimePipeline


# pylint: disable=protected-access
# pylint: disable=unused-variable
@register(name='add_operator')
class AddOperator:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor + x


@register
def sub_operator(x, y):
    return x - y

def func(x):
    return x + 2


class TestPipeline(unittest.TestCase):
    """
    Test Pipeline
    """
    def test_property(self):
        pipe1 = Pipeline.input('a', 'b').map('a', 'c', ops.towhee.test_op1('a'))
        self.assertEqual(len(pipe1.dag), 2)
        self.assertEqual(pipe1.config, None)

        pipe2 = pipe1.set_config({'parallel': 4}).map('c', 'd', ops.test_op3())
        self.assertEqual(len(pipe2.dag), 3)
        self.assertEqual(pipe2.config, {'parallel': 4})

        pipe3 = pipe1.map('a', 'c', ops.towhee.test_op3('a')).set_config({'parallel': 3})
        self.assertEqual(len(pipe3.dag), 3)
        self.assertEqual(pipe3.config, {'parallel': 3})

        with self.assertRaises(ValueError):
            pipe2.concat(pipe3)
        pipe3.set_config({'parallel': 4})
        pipe4 = pipe2.concat(pipe3)
        self.assertEqual(len(pipe4.dag), 5)
        self.assertEqual(pipe4.config, {'parallel': 4})

        pipe3.set_config(None)
        pipe4 = pipe2.concat(pipe3)
        self.assertEqual(len(pipe4.dag), 5)
        self.assertEqual(pipe4.config, {'parallel': 4})

    def test_raise(self):
        with self.assertRaises(ValueError):
            Pipeline.input('a').map('a', 'c', 12).output('c')

    def test_callable(self):
        pipe = (Pipeline.input('a')
                .map('a', 'c', func)
                .output('c'))
        res = pipe(1).get()
        self.assertEqual(res[0], 3)

    def test_lambda(self):
        pipe = (Pipeline.input('a')
                .map('a', 'c', lambda x: x+2)
                .output('c'))
        res = pipe(2).get()
        self.assertEqual(res[0], 4)

    # TODO: Add pipeline config test
    def test_local(self):
        pipe1 = (Pipeline.input('a')
                 .set_config({'parallel': 3})
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

    # TODO: Add more test with flat_map, filter, time_window, window, window_all
    def test_flat_map(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.flat_map('a', 'd', ops.test_op1())
        self.assertEqual(pipe1.dag[pipe1._clo_node]['iter_info']['type'], 'flat_map')

    def test_filter(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.filter('a', 'd', 'c', ops.test_op1())
        self.assertEqual(pipe1.dag[pipe1._clo_node]['iter_info']['type'], 'filter')

    def test_time_window(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.time_window('a', 'd', 'c', 2, 2, ops.test_op1())
        self.assertEqual(pipe1.dag[pipe1._clo_node]['iter_info']['type'], 'time_window')

    def test_window(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.window('a', 'd', 2, 2, ops.test_op1())
        self.assertEqual(pipe1.dag[pipe1._clo_node]['iter_info']['type'], 'window')

    def test_window_all(self):
        pipe0 = Pipeline.input('a', 'b', 'c')
        pipe1 = pipe0.window_all('a', 'd', ops.test_op1())
        self.assertEqual(pipe1.dag[pipe1._clo_node]['iter_info']['type'], 'window_all')
