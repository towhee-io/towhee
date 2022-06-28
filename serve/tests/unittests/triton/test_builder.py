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

from serve.triton.builder import Builder


towhee_dag = {}

triton_dag = {
    0: {
        'id': 0,
        'mode_name': 'image_decode',
        'model_version': 1,
        'inputs': {
            'INPUT0': ('TYPE_STRING', []),
        },
        'outputs': {
            'OUTPUT0': ('TYPE_INT8', [-1, -1, 3]),
            'OUTPUT1': ('TYPE_STRING', []),
        },
        'child_ids': [1]
    },
    1: {
        'id': 1,
        'mode_name': 'clip_preprocess',
        'model_version': 1,
        'inputs': {
            'IUTPUT0': ('TYPE_INT8', [-1, -1, 3]),
            'IUTPUT1': ('TYPE_STRING', []),            
        },
        'outputs': {
            'OUTPUT0': ('TYPE_FP32', [-1, 3, 224, 224])
        },
        'child_ids': [2]
    },
    2: {
        'id': 2,
        'mode_name': 'clip_model',
        'model_version': 1,
        'inputs': {
            'IUTPUT0': ('TYPE_FP32', [-1, 3, 224, 224])
        },
        'outputs': {
            'OUTPUT0': ('TYPE_FP32', [-1, 512])
        },
        'child_ids': [3]        
    },
    3: {
        'id': 3,
        'mode_name': 'clip_postprocess',
        'model_version': 1,
        'inputs': {
            'IUTPUT0': ('TYPE_FP32', [-1, 512])
        },
        'outputs': {
            'OUTPUT0': ('TYPE_FP32', [512])
        },
        'child_ids': []
    }
}


class TestBuilder(unittest.TestCase):
    def test_builder(self):
        test_dag = {'start': {'op': 'stream', 'op_name': 'dummy_input', 'is_stream': False, 'init_args': None, 'call_args': {'*arg': (), '*kws': {}}, 'parent_ids': [], 'child_ids': ['cb2876f3']}, 'cb2876f3': {'op': 'map', 'op_name': 'local/triton_py', 'is_stream': True, 'init_args': {}, 'call_args': {'*arg': None, '*kws': {}}, 'parent_ids': ['start'], 'child_ids': ['fae9ba13']}, 'fae9ba13': {'op': 'map', 'op_name': 'local/triton_nnop', 'is_stream': True, 'init_args': {'model_name': 'test'}, 'call_args': {'*arg': None, '*kws': {}}, 'parent_ids': ['cb2876f3'], 'child_ids': ['end']}, 'end': {'op': 'end', 'op_name': 'end', 'init_args': None, 'call_args': None, 'parent_ids': ['fae9ba13'], 'child_ids': []}}
        root = './root'
        builer = Builder(test_dag, root, [])
        assert builer.load() is True
        print(builer._runtime_dag)
        self.assertTrue(builer.build())
