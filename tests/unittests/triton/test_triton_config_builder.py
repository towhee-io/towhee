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

from towhee.serve.triton.triton_config_builder import create_modelconfig

from . import EXPECTED_FILE_PATH


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
