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
from shutil import rmtree
import os

from towhee.hub.operator_manager import OperatorManager

public_path = Path(__file__).parent.parent.resolve()


@unittest.skip('Not pass')
class TestOperatorManager(unittest.TestCase):
    """
    Unittest for OperatorManager.
    """
    def test_init(self):
        nn_file_src = public_path / 'mock_operators' / 'nnoperator_template'
        py_file_src = public_path / 'mock_operators' / 'pyoperator_template'
        nn_file_dst = public_path / 'test_cache' / 'operators' / 'towhee' / 'ci_test_nnoperator' / 'main'
        py_file_dst = public_path / 'test_cache' / 'operators' / 'towhee' / 'ci_test_pyoperator' / 'main'

        if nn_file_dst.is_dir():
            rmtree(nn_file_dst)
        if py_file_dst.is_dir():
            rmtree(py_file_dst)

        nn_rm = OperatorManager('towhee', 'ci-test-operator')
        py_rm = OperatorManager('towhee', 'ci-test-operator')

        nn_rm.init(True, nn_file_src, nn_file_dst)
        py_rm.init(False, py_file_src, py_file_dst)

        self.assertIn('ci_test_operator.py', os.listdir(nn_file_dst))
        self.assertIn('ci_test_operator.yaml', os.listdir(nn_file_dst))
        self.assertIn('pytorch', os.listdir(nn_file_dst))

        self.assertIn('ci_test_operator.py', os.listdir(py_file_dst))
        self.assertIn('ci_test_operator.yaml', os.listdir(py_file_dst))
        self.assertNotIn('pytorch', os.listdir(py_file_dst))

        rmtree(nn_file_dst.parent)
        rmtree(py_file_dst.parent)
