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

from towhee.runtime.pipeline_manager import PipelineManager


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
                'type': 'local',
                'init_args': (),
                'init_kws': {},
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
                'type': 'local',
                'init_args': (),
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
        pipeline_manager = PipelineManager(self.dag_dict)
        result1 = pipeline_manager(1, 2, 3)
        self.assertEqual(result1[0].diff, -1)
        self.assertEqual(result1[1].sum, 13)

        result2 = pipeline_manager(2, 2, -10)
        self.assertEqual(result2[0].diff, 0)
        self.assertEqual(result2[1].sum, 0)

    def test_preload(self):
        # pylint: disable=protected-access
        """
        _input(map)[(a, b, c)]->sub_op(map)[(a, b)-(d,)]->add_op(map)[(c,)-(e,)]->_output(map)[(d, e)]
        """
        pipeline_manager = PipelineManager(self.dag_dict)
        self.assertEqual(len(pipeline_manager._operator_pool._all_ops), 0)
        pipeline_manager.preload()
        self.assertEqual(len(pipeline_manager._operator_pool._all_ops), 4)
