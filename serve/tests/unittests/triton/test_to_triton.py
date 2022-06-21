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
from towhee import ops
from serve.triton.to_triton_models import PyOpToTriton, ProcessToTriton


class TestPyOpToTriton(unittest.TestCase):
    def _test_case(self, root_dir):
        op = ops.local.triton_py().get_op()
        to_triton = PyOpToTriton(op, root_dir, 'py_to_triton_test', 'local', 'triton_py', {})
        to_triton.to_triton()
    
    def test_py_to_triton(self):
        self._test_case('./')


class TestProcessor(unittest.TestCase):
    def test_processor(self):
        op = ops.local.triton_nnop(model_name='test').get_op()
        to_triton = ProcessToTriton(op, './', 'preprocess', 'preprocess', 'triton_nnop')
        to_triton.to_triton()
