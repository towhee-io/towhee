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
                'e4b074eb': {
                    'op': 'map',
                    'op_name': 'local/triton_py',
                    'is_stream': True,
                    'init_args': {},
                    'call_args': {
                        '*arg': (),
                        '*kws': {}
                    },
                    'op_config': {'device_ids': [1]},
                    'input_info': [('start', 'path')],
                    'output_info': ['img', 'select'],
                    'parent_ids': ['start'],
                    'child_ids': ['db5377c3']
                },
                'db5377c3': {
                    'op': 'map',
                    'op_name': 'local/triton_nnop',
                    'is_stream': True,
                    'init_args': {
                        'model_name': 'test',
                    },
                    'call_args': {
                        '*arg': (),
                        '*kws': {}
                    },
                    'op_config': {
                        'device_ids': [1, 2],
                        'dynamic_batching': {
                            'max_batch_size': 128,
                            'preferred_batch_size': [1, 2],
                            'preferred_max_queue_delay_microseconds': 10000
                        }
                    },
                    'input_info': [('e4b074eb', 'img')],
                    'output_info': ['vec'],
                    'parent_ids': ['e4b074eb'],
                    'child_ids': ['end']
                },
                'end': {
                    'op': 'end',
                    'op_name': 'end',
                    'init_args': None,
                    'call_args': None,
                    'op_config': None,
                    'input_info': None,
                    'output_info': None,
                    'parent_ids': ['db5377c3'],
                    'child_ids': []
                },
                'start': {
                    'op': 'stream',
                    'op_name': 'dummy_input',
                    'is_stream': False,
                    'init_args': None,
                    'call_args': {
                        '*arg': (),
                        '*kws': {}
                    },
                    'op_config': None,
                    'input_info': None,
                    'output_info': None,
                    'parent_ids': [],
                    'child_ids': ['e4b074eb']
                }
            }
        with TemporaryDirectory(dir='./') as root:
            builer = Builder(test_dag, root, ['tensorrt'])
            self.assertTrue(builer.build())

            expect_root = Path(EXPECTED_FILE_PATH) / 'ensemble'
            dst = Path(root) / 'pipeline'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')

            expect_root = Path(EXPECTED_FILE_PATH) / 'py_to_triton_test'
            dst = Path(root) / 'e4b074eb_local_triton_py'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            self.assertTrue(filecmp.cmp(expect_root / '1' / 'model.py', dst / '1' / 'model.py'))

            expect_root = Path(EXPECTED_FILE_PATH) / 'preprocess'
            dst = Path(root) / 'db5377c3_local_triton_nnop_preprocess'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            pk = dst / '1' / 'preprocess.pickle'
            m_file = dst / '1' / 'model.py'
            self.assertTrue(pk.is_file())
            self.assertTrue(m_file.is_file())

            expect_root = Path(EXPECTED_FILE_PATH) / 'postprocess'
            dst = Path(root) / 'db5377c3_local_triton_nnop_postprocess'
            self.assertTrue(filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt'))
            pk = dst / '1' / 'postprocess.pickle'
            m_file = dst / '1' / 'model.py'
            self.assertTrue(pk.is_file())
            self.assertTrue(m_file.is_file())

            expect_root = Path(EXPECTED_FILE_PATH) / 'nnop'
            dst = Path(root) / 'db5377c3_local_triton_nnop_model'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')

            expect_root = Path(EXPECTED_FILE_PATH) / 'ensemble'
            dst = Path(root) / 'pipeline'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')

    def test_old_version_nnop(self):
        test_dag = {
            'start': {
                'op_name': 'dummy_input', 'init_args': None, 'child_ids': ['cb2876f3'], 'input_info': None, 'output_info': None, 'parent_ids': []
            },
            'cb2876f3': {
                'op_name': 'local/trion_nnop_oldversion', 'init_args': {}, 'child_ids': ['end'], 'input_info': None, 'output_info': None, 'parent_ids':['start']
            },
            'end': {
                'op_name': 'end', 'init_args': None, 'call_args': None, 'child_ids': [], 'input_info': None, 'output_info': None, 'parent_ids':['cb2876f3']
            }
        }
        with TemporaryDirectory(dir='./') as root:
            builer = Builder(test_dag, root, ['tensorrt'])
            self.assertTrue(builer.build())
            dst = Path(root) / 'cb2876f3_local_trion_nnop_oldversion' / '1' / 'model.py'
            self.assertTrue(dst.is_file())

    def test_schema_builder(self):
        test_dag = {
                'start': {
                    'op': 'stream',
                    'op_name': 'dummy_input',
                    'is_stream': False,
                    'init_args': None,
                    'call_args': {
                        '*arg': (),
                        '*kws': {}
                    },
                    'op_config': None,
                    'input_info': None,
                    'output_info': None,
                    'parent_ids': [],
                    'child_ids': ['e4b074eb']
                },
                'e4b074eb': {
                    'op': 'map',
                    'op_name': 'local/triton_py',
                    'is_stream': True,
                    'init_args': {},
                    'call_args': {
                        '*arg': (),
                        '*kws': {}
                    },
                    'op_config': {'device_ids': [1]},
                    'input_info': [('start', 'path')],
                    'output_info': ['img', 'select'],
                    'parent_ids': ['start'],
                    'child_ids': ['db5377c3']
                },
                'db5377c3': {
                    'op': 'map',
                    'op_name': 'local/triton_nnop',
                    'is_stream': True,
                    'init_args': {
                        'model_name': 'test',
                    },
                    'call_args': {
                        '*arg': (),
                        '*kws': {}
                    },
                    'op_config': {
                        'device_ids': [1, 2],
                        'dynamic_batching': {
                            'max_batch_size': 128,
                            'preferred_batch_size': [1, 2],
                            'preferred_max_queue_delay_microseconds': 10000
                        }
                    },
                    'input_info': [('e4b074eb', 'img'), ('e4b074eb', 'select')],
                    'output_info': ['vec'],
                    'parent_ids': ['e4b074eb'],
                    'child_ids': ['end']
                },
                'a5c012ac': {
                    'op': 'map',
                    'op_name': 'local/triton_test_py',
                    'is_stream': True,
                    'init_args': {},
                    'call_args': {
                        '*arg': (),
                        '*kws': {}
                    },
                    'op_config': {'device_ids': [2]},
                    'input_info': [('start', 'path'), ('db5377c3', 'vec')],
                    'output_info': ['obj'],
                    'parent_ids': ['start', 'db5377c3'],
                    'child_ids': ['qf0239us']
                },
                'qf0239us': {
                    'op': 'map',
                    'op_name': 'local/triton_test2_py',
                    'is_stream': True,
                    'init_args': {},
                    'call_args': {
                        '*arg': (),
                        '*kws': {}
                    },
                    'op_config': {},
                    'input_info': [('a5c012ac', 'obj')],
                    'output_info': ['obj'],
                    'parent_ids': ['a5c012ac'],
                    'child_ids': ['end']
                },
                'end': {
                    'op': 'end',
                    'op_name': 'end',
                    'init_args': None,
                    'call_args': None,
                    'op_config': None,
                    'input_info': None,
                    'output_info': None,
                    'parent_ids': ['qf0239us'],
                    'child_ids': []
                }      
            }
        with TemporaryDirectory(dir='./') as root:
            builer = Builder(test_dag, root, ['tensorrt'])
            self.assertTrue(builer.build())
            expect_root = Path(EXPECTED_FILE_PATH) / 'schema'
            dst = Path(root) / 'pipeline'
            filecmp.cmp(expect_root / 'config.pbtxt', dst / 'config.pbtxt')