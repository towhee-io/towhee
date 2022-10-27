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
from itertools import zip_longest

from towhee.runtime.runtime_pipeline import RuntimePipeline
from towhee.runtime.data_queue import Empty


class TestWindowPipeline(unittest.TestCase):
    """
    Test pipelines contain different kinds of node.
    """
    dag_dict = {
        '_input': {
            'inputs': ('a',),
            'outputs': ('a',),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': ['op0']
        },
        'op0': {
                'inputs': ('a',),
                'outputs': ('b',),
                'iter_info': {
                    'type': 'flat_map',
                    'param': None
                },
                'op_info': {
                    'operator': lambda x: x,
                    'type': 'lambda',
                    'init_args': None,
                    'init_kws': None,
                    'tag': 'main'
                },
                'config': None,
                'next_nodes': ['op1']
        },
        'op1': {
            'inputs': ('b',),
            'outputs': ('c',),
            'op_info': {
                'type': 'lambda',
                'operator': sum,
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'window',
                'param': {
                    'size': 1,
                    'step': 1
                }
            },
            'config': {},
            'next_nodes': ['_output']
        },
        '_output': {
            'inputs': ('b', 'c'),
            'outputs': ('b', 'c'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': None
        },
    }

    def test_normal(self):
        size = 1
        step = 1
        self.dag_dict['op1']['iter_info']['param']['size'] = size
        self.dag_dict['op1']['iter_info']['param']['step'] = step
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        result = runtime_pipeline([1, 1, 1, 1, 1, 1])
        for _ in range(6):
            self.assertEqual(result.get(), [1, 1])

    def test_zero_size(self):
        with self.assertRaises(ValueError):
            size = 0
            step = 1
            self.dag_dict['op1']['iter_info']['param']['size'] = size
            self.dag_dict['op1']['iter_info']['param']['step'] = step
            runtime_pipeline = RuntimePipeline(self.dag_dict)
            runtime_pipeline([1, 1, 1, 1, 1, 1])

    def test_zero_step(self):
        with self.assertRaises(ValueError):
            size = 1
            step = 0
            self.dag_dict['op1']['iter_info']['param']['size'] = size
            self.dag_dict['op1']['iter_info']['param']['step'] = step
            runtime_pipeline = RuntimePipeline(self.dag_dict)
            runtime_pipeline([1, 1, 1, 1, 1, 1])

    def test_step_size_combination(self):
        for step in [1, 2, 3, 4, 5, 6, 7]:
            for size in [1, 2, 3, 4, 5, 6, 7]:
                self.dag_dict['op1']['iter_info']['param']['size'] = size
                self.dag_dict['op1']['iter_info']['param']['step'] = step
                runtime_pipeline = RuntimePipeline(self.dag_dict)
                result = runtime_pipeline([1, 1, 1, 1, 1, 1])
                for i in range(6):
                    self.assertEqual(result.get(), [1, size if i*step + size - 1  < 6 else 6 - i*step if i*step < 6 else Empty()])

    def test_zero_data(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        result = runtime_pipeline([])
        self.assertEqual(result.size, 0)

class TestWindowAllPipeline(unittest.TestCase):
    """
    Test pipelines contain different kinds of node.
    """
    dag_dict = {
        '_input': {
            'inputs': ('a',),
            'outputs': ('a',),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': ['op0']
        },
        'op0': {
                'inputs': ('a',),
                'outputs': ('b',),
                'iter_info': {
                    'type': 'flat_map',
                    'param': None
                },
                'op_info': {
                    'operator': lambda x: x,
                    'type': 'lambda',
                    'init_args': None,
                    'init_kws': None,
                    'tag': 'main'
                },
                'config': None,
                'next_nodes': ['op1']
        },
        'op1': {
            'inputs': ('b',),
            'outputs': ('c',),
            'op_info': {
                'type': 'lambda',
                'operator': sum,
                'tag': 'main',
                'init_args': None,
                'init_kws': {}
            },
            'iter_info': {
                'type': 'window_all',
                'param': {}
            },
            'config': {},
            'next_nodes': ['_output']
        },
        '_output': {
            'inputs': ('b', 'c'),
            'outputs': ('b', 'c'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': None
        },
    }

    def test_normal(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        result = runtime_pipeline([1, 1, 1, 1, 1, 1])
        for i in range(6):
            self.assertEqual(result.get(), [1, 6 if i == 0 else Empty()])

    def test_zero_data(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        result = runtime_pipeline([])
        self.assertEqual(result.size, 0)


def split(x, y):
    for i, j in zip_longest(x, y):
        yield i if i else 0, j if j else 0


class TestTimeWindowPipeline(unittest.TestCase):
    """
    Test pipelines contain different kinds of node.
    """
    dag_dict = {
        '_input': {
            'inputs': ('a', 'timestamp'),
            'outputs': ('a', 'timestamp'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': ['op0']
        },
        'op0': {
                'inputs': ('a', 'timestamp'),
                'outputs': ('b', 'timestamp'),
                'iter_info': {
                    'type': 'flat_map',
                    'param': None
                },
                'op_info': {
                    'operator': split,
                    'type': 'lambda',
                    'init_args': None,
                    'init_kws': None,
                    'tag': 'main'
                },
                'config': None,
                'next_nodes': ['op1']
        },
        'op1': {
            'inputs': ('b',),
            'outputs': ('c',),
            'op_info': {
                'type': 'lambda',
                'operator': sum,
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
        },
        '_output': {
            'inputs': ('b', 'c', 'timestamp'),
            'outputs': ('b', 'c', 'timestamp'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': None
        },
    }

    def test_normal(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        result = runtime_pipeline([1, 1, 1, 1, 1, 1], [0, 1000, 2000, 3000, 4000, 5000])
        for i in range(6):
            self.assertEqual(result.get(), [1, 3 if i < 2 else Empty(), i*1000])

    def test_zero_data(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        result = runtime_pipeline([], [0, 1000, 2000, 3000, 4000, 5000])
        for i in range(6):
            self.assertEqual(result.get(), [0, 0 if i < 2 else Empty(), i*1000])

    def test_zero_timestamp(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        result = runtime_pipeline([1, 1, 1, 1, 1, 1], [])
        for i in range(6):
            self.assertEqual(result.get(), [1, 6 if i < 1 else Empty(), 0])

    def test_uneven_timestamp(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        timestamp = [0, 500, 2500, 4000, 6000, 7000]
        result = runtime_pipeline([1, 1, 1, 1, 1, 1], timestamp)
        for i in range(6):
            self.assertEqual(result.get(), [1, 3 if i == 0 else 1 if i == 1 else 2 if i == 2 else Empty(), timestamp[i]])

    def test_less_data(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        timestamp = [0, 1000, 2000, 3000, 4000, 5000]
        result = runtime_pipeline([1, 1, 1, 1], timestamp)
        for i in range(6):
            self.assertEqual(result.get(), [1 if i < 4 else 0, 3 if i == 0 else 1 if i ==1 else Empty(), timestamp[i]])

    def test_less_timestampe(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        timestamp = [0, 1000, 2000, 3000]
        result = runtime_pipeline([1, 1, 1, 1, 1, 1], timestamp)
        for i in range(6):
            self.assertEqual(result.get(), [1, 3 if i == 0 else 1 if i == 1 else Empty(), timestamp[i] if i < 4 else 0])

    def test_unordered_timestampe(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        timestamp = [2000, 4000, 5000, 8000, 3000, 4000]
        result = runtime_pipeline([1, 1, 1, 1, 1, 1], timestamp)
        for i in range(6):
            self.assertEqual(result.get(), [1, 1 if i in (0, 2) else 2 if i == 1 else Empty(), timestamp[i]])

        timestamp = [1000, 500, 3000, 2500, 7000, 4000]
        result = runtime_pipeline([1, 1, 1, 1, 1, 1], timestamp)
        for i in range(6):
            self.assertEqual(result.get(), [1, 2 if i == 0 else 1 if i in (1, 2) else Empty(), timestamp[i]])

    def test_zero_range(self):
        with self.assertRaises(ValueError):
            dag_dict = copy.deepcopy(self.dag_dict)
            dag_dict['op1']['iter_info']['param']['time_range_sec'] = 0
            runtime_pipeline = RuntimePipeline(dag_dict)
            runtime_pipeline([1, 1, 1, 1, 1, 1], [0, 1000, 2000, 3000, 4000, 5000])

    def test_zero_step(self):
        with self.assertRaises(ValueError):
            dag_dict = copy.deepcopy(self.dag_dict)
            dag_dict['op1']['iter_info']['param']['time_step_sec'] = 0
            runtime_pipeline = RuntimePipeline(dag_dict)
            runtime_pipeline([1, 1, 1, 1, 1, 1], [0, 1000, 2000, 3000, 4000, 5000])

    def test_step_range_combination(self):
        dag_dict = copy.deepcopy(self.dag_dict)
        for step in [1, 2, 3, 5, 6, 7]:
            for size in [1, 2, 3, 5, 6, 7]:
                dag_dict['op1']['iter_info']['param']['time_step_sec'] = step
                dag_dict['op1']['iter_info']['param']['time_range_sec'] = size
                runtime_pipeline = RuntimePipeline(dag_dict)
                timestamp = [0, 1000, 2000, 3000, 4000, 5000]
                result = runtime_pipeline([1, 1, 1, 1, 1, 1], timestamp)
                for i in range(6):
                    self.assertEqual(result.get(), [1, size if i*step + size - 1  < 6 else 6 - i*step if i*step < 6 else Empty(), timestamp[i]])
