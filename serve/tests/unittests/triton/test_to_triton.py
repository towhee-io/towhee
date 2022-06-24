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
from serve.triton.to_triton_models import PyOpToTriton, PreprocessToTriton, PostprocessToTriton,ModelToTriton

from . import EXPECTED_FILE_PATH


class TestPyOpToTriton(unittest.TestCase):
    '''
    Test pyop to triton.
    '''

    def test_py_to_triton(self):
        with TemporaryDirectory(dir='./') as root:
            op = ops.local.triton_py().get_op()
            to_triton = PyOpToTriton(op, root, 'py_to_triton_test', 'local', 'triton_py', {})
            to_triton.to_triton()
            expect_root = Path(EXPECTED_FILE_PATH) / 'py_to_triton_test'
            dst = Path(root) / 'py_to_triton_test'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            self.assertTrue(filecmp.cmp(expect_root / '1' / 'model.py', dst / '1' / 'model.py'))


class TestPreprocessor(unittest.TestCase):
    '''
    Test nnop process to triton
    '''

    def test_processor(self):
        with TemporaryDirectory(dir='./') as root:
            op = ops.local.triton_nnop(model_name='test').get_op()
            to_triton = PreprocessToTriton(op, root, 'preprocess')
            to_triton.to_triton()

            expect_root = Path(EXPECTED_FILE_PATH) / 'preprocess'
            dst = Path(root) / 'preprocess'
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
            to_triton = PostprocessToTriton(op, root, 'postprocess')
            to_triton.to_triton()

            expect_root = Path(EXPECTED_FILE_PATH) / 'postprocess'
            dst = Path(root) / 'postprocess'
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
            to_triton = ModelToTriton(op.model, root, 'nnop')
            to_triton.to_triton()
            expect_root = Path(EXPECTED_FILE_PATH) / 'nnop'
            dst = Path(root) / 'nnop'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')
