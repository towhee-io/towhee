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
from pathlib import Path
import filecmp
from tempfile import TemporaryDirectory

from towhee.serve.triton.builder import Builder
from . import EXPECTED_FILE_PATH


class TestBuilder(unittest.TestCase):
    '''
    Test triton buidler.
    '''
    def test_builder(self):
        test_dag = {
            'start': {
                'op_name': 'dummy_input', 'init_args': None, 'child_ids': ['cb2876f3']
            },
            'cb2876f3': {
                'op_name': 'local/triton_py', 'init_args': {}, 'child_ids': ['fae9ba13']
            },
            'fae9ba13': {
                'op_name': 'local/triton_nnop', 'init_args': {'model_name': 'test'},'child_ids': ['end']
            },
            'end': {
                'op_name': 'end', 'init_args': None, 'call_args': None, 'child_ids': []
            }
        }
        with TemporaryDirectory(dir='./') as root:
            builer = Builder(test_dag, root, ['tensorrt'])
            self.assertTrue(builer.build())

            expect_root = Path(EXPECTED_FILE_PATH) / 'ensemble'
            dst = Path(root) / 'pipeline'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')

            expect_root = Path(EXPECTED_FILE_PATH) / 'py_to_triton_test'
            dst = Path(root) / 'cb2876f3_local_triton_py'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            self.assertTrue(filecmp.cmp(expect_root / '1' / 'model.py', dst / '1' / 'model.py'))

            expect_root = Path(EXPECTED_FILE_PATH) / 'preprocess'
            dst = Path(root) / 'fae9ba13_local_triton_nnop_preprocess'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            pk = dst / '1' / 'preprocess.pickle'
            m_file = dst / '1' / 'model.py'
            self.assertTrue(pk.is_file())
            self.assertTrue(m_file.is_file())

            expect_root = Path(EXPECTED_FILE_PATH) / 'postprocess'
            dst = Path(root) / 'fae9ba13_local_triton_nnop_postprocess'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            pk = dst / '1' / 'postprocess.pickle'
            m_file = dst / '1' / 'model.py'
            self.assertTrue(pk.is_file())
            self.assertTrue(m_file.is_file())

            expect_root = Path(EXPECTED_FILE_PATH) / 'nnop'
            dst = Path(root) / 'fae9ba13_local_triton_nnop_model'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')

            expect_root = Path(EXPECTED_FILE_PATH) / 'ensemble'
            dst = Path(root) / 'pipeline'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')

    def test_old_version_nnop(self):
        test_dag = {
            'start': {
                'op_name': 'dummy_input', 'init_args': None, 'child_ids': ['cb2876f3']
            },
            'cb2876f3': {
                'op_name': 'local/trion_nnop_oldversion', 'init_args': {}, 'child_ids': ['end']
            },
            'end': {
                'op_name': 'end', 'init_args': None, 'call_args': None, 'child_ids': []
            }
        }
        with TemporaryDirectory(dir='./') as root:
            builer = Builder(test_dag, root, ['tensorrt'])
            self.assertTrue(builer.build())
            dst = Path(root) / 'cb2876f3_local_trion_nnop_oldversion' / '1' / 'model.py'
            self.assertTrue(dst.is_file())
