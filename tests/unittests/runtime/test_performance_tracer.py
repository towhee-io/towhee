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
import json
import unittest
from pathlib import Path

from towhee import pipe
from towhee.runtime.factory import ops, register
from towhee.runtime.performance_profiler import TimeProfiler, PerformanceProfiler


public_path = Path(__file__).parent.parent.resolve()


@register
def flat_map(x, y):
    return list(zip(x, y))


@register
def cal(x, y):
    return x * y


@register(name='add_operator')
class AddOperator:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor + x


# pylint: disable=protected-access
class TestPerformanceProfiler(unittest.TestCase):
    """
    Unit test for PerformanceProfiler.
    """
    def test_check(self):
        p = pipe.input('a').output('a')
        _ = p.debug(1, profiler=True)
        time_profiler = p._time_profiler_list[0]
        time_profiler.record('_run_pipe', 'pipe_out')
        with self.assertRaises(BaseException):
            _ = PerformanceProfiler(time_profiler.time_record, p._dag_repr)

    def test_tracer_concat(self):
        pipe0 = pipe.input('a', 'b', 'c')
        pipe1 = pipe0.map('a', 'd', ops.add_operator(10))
        pipe2 = pipe0.map(('b', 'c'), 'e', ops.cal())
        pipe3 = pipe0.map(('b', 'c'), 'f', ops.cal())
        pipe4 = pipe3.concat(pipe1, pipe2).output('d', 'e', 'f')
        pipe4.debug(1, 2, 3, profiler=True).get()
        pipe4.debug(2, 2, 3, profiler=True).get()
        pipe4.debug(8, 2, 3, profiler=True).get()

        pp = pipe4.profiler()
        self.assertEqual(len(pp.pipes_profiler), 3)
        self.assertEqual(len(pp.node_report), 6)
        pp.sort()[0].show()
        pp.max().show()

    def test_tracer_flat_map(self):
        pipe0 = (pipe.input('d')
                 .flat_map('d', ('n1', 'n2'), lambda x: ((a, b) for a, b in x))
                 .output('n1', 'n2'))
        data = [(i, i + 1) for i in range(10)]
        pipe0.debug(data, profiler=True)

        pp = pipe0.profiler()
        self.assertEqual(len(pp.pipes_profiler), 1)
        self.assertEqual(len(pp.node_report), 3)
        pp.show()

    def test_tracer_time_window(self):
        pipe0 = (pipe.input('d')
                 .flat_map('d', ('n1', 'n2', 't'), lambda x: ((a, b, c) for a, b, c in x))
                 .time_window(('n1', 'n2'), ('s1', 's2'), 't', 10, 5, ops.local.sum2())
                 .output('s1', 's2'))
        data = [(i, i + 1, i * 1000) for i in range(100) if i < 3 or i > 91]
        pipe0.debug(data, profiler=True)

        pp = pipe0.profiler()
        self.assertEqual(len(pp.pipes_profiler), 1)
        self.assertEqual(len(pp.node_report), 4)
        pp.show()

        pipe0.reset_profiler()  # reset and warning
        pipe0.profiler()

    def test_tracer_window_all(self):
        pipe0 = (pipe.input('n1', 'n2')
                 .flat_map(('n1', 'n2'), ('n1', 'n2'), lambda x, y: list(zip(x, y)))
                 .map('n1', 'n2', lambda x: x+1)
                 .window_all(('n1', 'n2'), ('s1', 's2'), ops.local.sum2())
                 .output('s1', 's2'))
        pipe0.debug([1, 2, 3, 4], [2, 3, 4, 5], profiler=True)
        pipe0.debug([2, 2, 3, 4], [3, 3, 4, 5], profiler=True)

        pp = pipe0.profiler()
        self.assertEqual(len(pp.pipes_profiler), 2)
        self.assertEqual(len(pp.node_report), 5)
        pp.show()

    def test_tracer_filter(self):
        def filter_func(x, y):
            return x > 10 and y > 5

        pipe0 = (pipe.input('a', 'b', 'c')
                 .map('c', 'c', lambda x: x + 1)
                 .filter('c', 'd', ('a', 'b'), filter_func)
                 .output('a', 'b', 'c', 'd'))
        pipe0.debug(5, 6, 7, profiler=True)
        pipe0.debug(15, 6, 7, profiler=True)

        pp = pipe0.profiler()
        self.assertEqual(len(pp.pipes_profiler), 2)
        self.assertEqual(len(pp.node_report), 4)
        pp.show()

        path1 = public_path / 'all_pipe.json'
        pp.dump(path1)
        with open(path1, encoding='utf-8') as f:
            json_tracing2 = json.load(f)
        self.assertEqual(len(json_tracing2), 78)

        path2 = public_path / 'pipe1.json'
        pp[0].dump(path2)
        with open(path2, encoding='utf-8') as f:
            json_tracing2 = json.load(f)
        self.assertEqual(len(json_tracing2), 35)

        path1.unlink()
        path2.unlink()

    def test_in_batch(self):
        pipe0 = (pipe.input('a', 'b', 'c')
                 .map('c', 'c', lambda x: x + 1)
                 .filter('c', 'd', ('a', 'b'), lambda x, y: x > 10)
                 .output('a', 'b', 'c', 'd'))
        pipe0.debug([(5, 6, 7)] * 10, batch=True, profiler=True)

        pp = pipe0.profiler()
        self.assertEqual(len(pp.pipes_profiler), 10)
        self.assertEqual(len(pp.node_report), 4)

class TestTimeProfiler(unittest.TestCase):
    """
    Unit test for TimeProfiler.
    """
    def test_profiler(self):
        tp = TimeProfiler()
        self.assertEqual(len(tp.time_record), 0)

    def test_record(self):
        tp = TimeProfiler()
        tp.record('no_op', 'in')
        self.assertEqual(len(tp.time_record), 0)

        tp.enable()
        tp.record('op1', 'init_in')
        tp.record('op1', 'init_out')
        self.assertEqual(len(tp.time_record), 2)

        tp.disable()
        tp.record('op1', 'init_in')
        self.assertEqual(len(tp.time_record), 2)

    def test_reset(self):
        tp = TimeProfiler()
        tp.enable()
        tp.record('no_op', 'in')
        self.assertEqual(len(tp.time_record), 1)

        tp.reset()
        self.assertEqual(len(tp.time_record), 0)
