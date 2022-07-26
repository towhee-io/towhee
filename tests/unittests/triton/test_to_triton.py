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
from tempfile import TemporaryDirectory
from pathlib import Path
import filecmp

from towhee import ops
from towhee.serve.triton.to_triton_models import PyOpToTriton, PreprocessToTriton, PostprocessToTriton, ModelToTriton, EnsembleToTriton

from . import EXPECTED_FILE_PATH


class TestPyOpToTriton(unittest.TestCase):
    '''
    Test pyop to triton.
    '''

    def test_py_to_triton(self):
        with TemporaryDirectory(dir='./') as root:
            op = ops.local.triton_py().get_op()
            to_triton = PyOpToTriton(op, root, 'cb2876f3_local_triton_py', 'local', 'triton_py', {}, {'device_ids': [1]})
            to_triton.to_triton()
            expect_root = Path(EXPECTED_FILE_PATH) / 'py_to_triton_test'
            dst = Path(root) / 'cb2876f3_local_triton_py'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            self.assertTrue(filecmp.cmp(expect_root / '1' / 'model.py', dst / '1' / 'model.py'))


class TestPreprocessor(unittest.TestCase):
    '''
    Test nnop process to triton
    '''

    def test_processor(self):
        with TemporaryDirectory(dir='./') as root:
            op = ops.local.triton_nnop(model_name='test').get_op()
            to_triton = PreprocessToTriton(op, root, 'fae9ba13_local_triton_nnop_preprocess',
                                           {
                                               'dynamic_batching': {
                                                   'max_batch_size': 128,
                                                   'preferred_batch_size': [1, 2],
                                                   'preferred_max_queue_delay_microseconds': 10000
                                               },
                                               'device_ids': [1, 2]
                                           })
            to_triton.to_triton()
            expect_root = Path(EXPECTED_FILE_PATH) / 'preprocess'
            dst = Path(root) / 'fae9ba13_local_triton_nnop_preprocess'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            pk = dst / '1' / 'preprocess.pickle'
            m_file = dst / '1' / 'model.py'
            self.assertTrue(pk.is_file())
            self.assertTrue(m_file.is_file())


class TestPostprocessor(unittest.TestCase):
    '''
    Test nnop process to triton
    '''

    def test_processor(self):
        with TemporaryDirectory(dir='./') as root:
            op = ops.local.triton_nnop(model_name='test').get_op()
            to_triton = PostprocessToTriton(op, root, 'fae9ba13_local_triton_nnop_postprocess',
                                            {
                                               'dynamic_batching': {
                                                   'max_batch_size': 128,
                                                   'preferred_batch_size': [1, 2],
                                                   'preferred_max_queue_delay_microseconds': 10000
                                               },
                                               'device_ids': [1, 2]
                                            })
            to_triton.to_triton()

            expect_root = Path(EXPECTED_FILE_PATH) / 'postprocess'
            dst = Path(root) / 'fae9ba13_local_triton_nnop_postprocess'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            pk = dst / '1' / 'postprocess.pickle'
            m_file = dst / '1' / 'model.py'
            self.assertTrue(pk.is_file())
            self.assertTrue(m_file.is_file())


class TestToModel(unittest.TestCase):
    '''
    Test nnop model to triton.
    '''

    def test_to_model(self):
        with TemporaryDirectory(dir='./') as root:
            op = ops.local.triton_nnop(model_name='test').get_op()
            to_triton = ModelToTriton(op, root, 'fae9ba13_local_triton_nnop_model', ['tensorrt'],
                                           {
                                               'device_ids': [1, 2],
                                               'instance_count': 2,
                                               'dynamic_batching': {
                                                   'max_batch_size': 128,
                                                   'preferred_batch_size': [1, 2],
                                                   'preferred_max_queue_delay_microseconds': 10000
                                               }
                                           })
            to_triton.to_triton()
            expect_root = Path(EXPECTED_FILE_PATH) / 'nnop'
            dst = Path(root) / 'fae9ba13_local_triton_nnop_model'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))


class TestToEnsemble(unittest.TestCase):
    '''
    Test create ensemble config.
    '''
    test_data = {
        'cb2876f3': {
            'id': 'cb2876f3', 'model_name': 'cb2876f3_local_triton_py', 'model_version': 1,
            'input': {'INPUT0': ('TYPE_STRING', [1])},
            'output': {'OUTPUT0': ('TYPE_INT8', [-1, -1, 3]), 'OUTPUT1': ('TYPE_STRING', [1])},
            'child_ids': ['fae9ba13']
        },
        'fae9ba13': {
            'id': 'fae9ba13',
            'model_name': 'fae9ba13_local_triton_nnop_preprocess', 'model_version': 1,
            'input': {'INPUT0': ('TYPE_INT8', [-1, -1, 3]), 'INPUT1': ('TYPE_STRING', [1])},
            'output': {'OUTPUT0': ('TYPE_FP32', [1, 3, 224, 224])},
            'child_ids': ['fae9ba13_local_triton_nnop_model']
        },
        'fae9ba13_local_triton_nnop_model': {
            'id': 'fae9ba13_local_triton_nnop_model',
            'model_name': 'fae9ba13_local_triton_nnop_model', 'model_version': 1,
            'input': {'INPUT0': ('TYPE_FP32', [-1, 3, 224, 224])},
            'output': {'OUTPUT0': ('TYPE_FP32', [-1, 512])},
            'child_ids': ['fae9ba13_local_triton_nnop_postprocess']
        },
        'fae9ba13_local_triton_nnop_postprocess': {
            'id': 'fae9ba13_local_triton_nnop_postprocess',
            'model_name': 'fae9ba13_local_triton_nnop_postprocess', 'model_version': 1,
            'input': {'INPUT0': ('TYPE_FP32', [-1, 512])},
            'output': {'OUTPUT0': ('TYPE_FP32', [512])},
            'child_ids': []
        }
    }

    def test_to_ensemble(self):
        with TemporaryDirectory(dir='./') as root:
            to_triton = EnsembleToTriton(TestToEnsemble.test_data, root, 'pipeline', 128)
            to_triton.to_triton()
            expect_root = Path(EXPECTED_FILE_PATH) / 'ensemble'
            dst = Path(root) / 'pipeline'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')
