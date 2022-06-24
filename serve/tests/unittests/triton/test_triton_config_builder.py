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
import numpy as np

from towhee._types.image import Image
from serve.triton.triton_config_builder import TritonModelConfigBuilder, create_ensemble, create_modelconfig

from . import EXPECTED_FILE_PATH


class TestTritonModelConfigBuilder(unittest.TestCase):
    """
    Unit test for TritonModelConfigBuilder.
    """

    # pylint: disable=protected-access

    def test_get_triton_schema(self):
        input_annotations = [(Image, (512, 512, 3)), (int, (1,)), (str, (1,))]
        output_annotations = [(np.float32, (1, 3, 224, 224))]

        input_schema = TritonModelConfigBuilder.get_input_schema(input_annotations)
        output_schema = TritonModelConfigBuilder.get_output_schema(output_annotations)
        expected_input_schema = {
            'INPUT0': ('TYPE_INT8', [512, 512, 3]),
            'INPUT1': ('TYPE_STRING', [1]),
            'INPUT2': ('TYPE_INT64', [1]),
            'INPUT3': ('TYPE_STRING', [1])
        }
        expected_output_schema = {'OUTPUT0': ('TYPE_FP32', [1, 3, 224, 224])}
        self.assertDictEqual(input_schema, expected_input_schema)
        self.assertDictEqual(output_schema, expected_output_schema)


class TestModelConfig(unittest.TestCase):
    '''
    Test to triton config.pbtxt
    '''

    def test_config(self):
        config = create_modelconfig('test_model', 128,
                                    {
                                        'INPUT0': ('TYPE_INT8', [-1, -1, 3]),
                                        'INPUT1': ('TYPE_FP32', [3, -1, -1]),
                                    }, {
                                        'OUTPUT0': ('TYPE_FP32', [-1, 512])
                                    }, 'python')

        e_f = EXPECTED_FILE_PATH + '/test_model.pbtxt'
        with open(e_f, 'rt', encoding='utf-8') as f:
            expect = f.read()
            self.assertEqual(config, expect)

class TestCreateEnsemble(unittest.TestCase):
    '''
    Test create ensenble
    '''
    def test_create_ensemblle(self):
        input_dag = {
            0: {
                'id': 0,
                'model_name': 'custom_zero_1_float32',
                'model_version': -1,
                'input': {
                    'INPUT0': ('TYPE_FP32', [16,1,2])
                },
                'output': {
                    'OUTPUT0': ('TYPE_FP32', [16])
                },
                'child_ids': [1]
            },
            1: {
                'id': 1,
                'model_name': 'custom_nobatch_zero_1_float32',
                'model_version': -1,
                'input': {
                    'INPUT0': ('TYPE_FP32', [16])
                },
                'output': {
                    'OUTPUT0': ('TYPE_FP32', [16])
                },
                'child_ids': [2]
            },
            2: {
                'id': 2,
                'model_name': 'graphdef_float32_float32_float32',
                'model_version': 1,
                'input': {
                    'INPUT0': ('TYPE_FP32', [16])
                },
                'output': {
                    'OUTPUT0': ('TYPE_FP32', [16,2,3,4]),
                    'OUTPUT1': ('TYPE_FP32', [16])
                },
                'child_ids': []
            }
        }
        res =  create_ensemble(input_dag, 'mix_nobatch_batch_float32_float32_float32', 8)
        e_f = EXPECTED_FILE_PATH + '/test_create_ensemble.pbtxt'
        with open(e_f, 'rt', encoding='utf-8') as f:
            expect = f.read()
            self.assertEqual(res, expect)
