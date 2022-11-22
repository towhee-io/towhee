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

import copy
import weakref
import unittest

from towhee.operator import PyOperator
from towhee.runtime.runtime_pipeline import RuntimePipeline
from towhee.runtime.operator_manager import OperatorRegistry

register = OperatorRegistry.register


class TestPipelineManager(unittest.TestCase):
    """
    PipelineManager Test
    """
    dag_dict = {
        '_input': {
            'inputs': ('a', 'b', 'c'),
            'outputs': ('a', 'b', 'c'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': ['op1']
        },
        'op1': {
            'inputs': ('a', 'b'),
            'outputs': ('d',),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'op_info': {
                'operator': 'local/sub_operator',
                'type': 'hub',
                'init_args': None,
                'init_kws': None,
                'tag': 'main',
            },
            'config': None,
            'next_nodes': ['op2']
        },
        'op2': {
            'inputs': ('c',),
            'outputs': ('e',),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'op_info': {
                'operator': 'local/add_operator',
                'type': 'hub',
                'init_args': None,
                'init_kws': {'factor': 10},
                'tag': 'main',
            },
            'config': None,
            'next_nodes': ['_output']
        },
        '_output': {
            'inputs': ('d', 'e'),
            'outputs': ('d', 'e'),
            'iter_info': {
                'type': 'map',
                'param': None
            },
            'next_nodes': None
        },
    }

    def test_run(self):
        """
        _input(map)[(a, b, c)]->sub_op(map)[(a, b)-(d,)]->add_op(map)[(c,)-(e,)]->_output(map)[(d, e)]
        """
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        result1 = runtime_pipeline(1, 2, 3).get()
        self.assertEqual(result1[0].diff, -1)
        self.assertEqual(result1[1].sum, 13)

        result2 = runtime_pipeline(2, 2, -10).get()
        self.assertEqual(result2[0].diff, 0)
        self.assertEqual(result2[1].sum, 0)

    def test_return_none(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['_output']['inputs'] = ()
        towhee_dag_test['_output']['outputs'] = ()
        runtime_pipeline = RuntimePipeline(towhee_dag_test)
        result = runtime_pipeline(1, 2, 3)
        self.assertEqual(result.size, 0)

    def test_preload(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        runtime_pipeline = RuntimePipeline(towhee_dag_test)
        runtime_pipeline.preload()
        self.assertEqual(len(runtime_pipeline._operator_pool._all_ops), 4) # pylint: disable=protected-access

    def test_raise_preload(self):
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['op2']['op_info']['init_kws'] = None
        runtime_pipeline = RuntimePipeline(towhee_dag_test)
        with self.assertRaises(RuntimeError):
            runtime_pipeline.preload()

    def test_raise_run(self):
        runtime_pipeline = RuntimePipeline(self.dag_dict)
        with self.assertRaises(RuntimeError):
            runtime_pipeline('a', 'b', 'c')

    def test_registry_with_pipe(self):
        # pylint: disable=unused-variable
        @register(name='test_rp/add_operator')
        class AddOperator(PyOperator):
            def __init__(self, fac):
                self.factor = fac

            def __call__(self, x):
                return self.factor + x

        @register(name='test_rp/sub_operator')
        def sub_operator(x, y):
            return x - y

        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['op1']['op_info']['operator'] = 'test-rp/sub-operator'
        towhee_dag_test['op2']['op_info']['operator'] = 'test-rp/add-operator'
        towhee_dag_test['op2']['op_info']['init_args'] = (1,)
        towhee_dag_test['op2']['op_info']['init_kws'] = None
        runtime_pipeline = RuntimePipeline(towhee_dag_test)
        result = runtime_pipeline(1, 2, 3).get()
        self.assertEqual(result, [-1, 4])

    def test_concat(self):
        """
        _input[(a, b, c)]->op1[(a, b)-(d,)]->op3(concat)->_output[(d, e)]
                    |------>op2[(c,)-(e,)]----^
        """
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['_input']['next_nodes'] = ['op1', 'op2']
        towhee_dag_test['op1']['next_nodes'] = ['op3']
        towhee_dag_test['op2']['next_nodes'] = ['op3']
        towhee_dag_test['op3'] = {
            'inputs': (),
            'outputs': (),
            'iter_info': {
                'type': 'concat',
                'param': None
            },
            'next_nodes': ['_output']
        }
        runtime_pipeline = RuntimePipeline(towhee_dag_test)
        result = runtime_pipeline(2, 2, -10).get()
        self.assertEqual(result[0].diff, 0)
        self.assertEqual(result[1].sum, 0)

    def test_concat_multi_output(self):
        """
        _input[(a, b, c)]->op1[(a, b)-(d,)]->op3(concat)->_output[(a, d, e)]
                    |------>op2[(c,)-(e,)]----^
        """
        towhee_dag_test = copy.deepcopy(self.dag_dict)
        towhee_dag_test['_input']['next_nodes'] = ['op1', 'op2']
        towhee_dag_test['op1']['next_nodes'] = ['op3']
        towhee_dag_test['op2']['next_nodes'] = ['op3']
        towhee_dag_test['op3'] = {
            'inputs': (),
            'outputs': (),
            'iter_info': {
                'type': 'concat',
                'param': None
            },
            'next_nodes': ['_output']
        }
        towhee_dag_test['_output']['inputs'] = ('a', 'd', 'e')
        towhee_dag_test['_output']['outputs'] = ('a', 'd', 'e')

        runtime_pipeline = RuntimePipeline(towhee_dag_test)
        result = runtime_pipeline(2, 2, -10).get()
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1].diff, 0)
        self.assertEqual(result[2].sum, 0)

    def test_runtime_pipeline_release(self):
        self.assertTrue('test-rp/sub-operator' in OperatorRegistry.op_names())
        num = 0
        pool_ref = callable
        # pylint: disable=unused-variable
        @register(name='add')
        class AddOperator(PyOperator):
            def __init__(self, fac):
                self.factor = fac

            def __call__(self, x):
                return self.factor + x

            def __del__(self):
                nonlocal num
                num += 1

        def func():
            towhee_dag_test = copy.deepcopy(self.dag_dict)
            towhee_dag_test['op1']['op_info']['operator'] = 'test-rp/sub-operator'
            towhee_dag_test['op2']['op_info']['operator'] = 'add'
            towhee_dag_test['op2']['op_info']['init_args'] = (1,)
            towhee_dag_test['op2']['op_info']['init_kws'] = None
            runtime_pipeline = RuntimePipeline(towhee_dag_test)
            nonlocal pool_ref
            pool_ref = weakref.ref(runtime_pipeline._operator_pool)  # pylint: disable=protected-access
            result = runtime_pipeline(1, 2, 3).get()
            self.assertEqual(result, [-1, 4])
            self.assertIsNotNone(pool_ref())

        for i in range(10):
            func()
            self.assertIsNone(pool_ref())
            self.assertEqual(num, i + 1)
