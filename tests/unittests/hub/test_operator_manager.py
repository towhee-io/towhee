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
from requests.exceptions import HTTPError

from towhee.utils.git_utils import GitUtils
from towhee.hub.operator_manager import OperatorManager



class TestOperatorManager(unittest.TestCase):
    """
    Unit test for OperatorManager.
    """
    def test_create(self):
        repo = 'repo-operator'
        manager = OperatorManager('towhee', repo)
        with self.assertRaises(HTTPError):
            manager.create('psw')

    def test_exists_create(self):
        repo = 'ci-test-operator'
        manager = OperatorManager('towhee', repo)
        manager.create('psw')

    def test_init_nnoperator(self):
        with TemporaryDirectory(dir='./') as root:
            public_path = Path(root)
            temp_repo = 'nnoperator-template'
            git = GitUtils(author='towhee', repo=temp_repo)
            git.clone(local_repo_path=public_path / 'test_cache' / temp_repo)

            nn_repo = 'nn-operator'
            nn_manager = OperatorManager('towhee', nn_repo)
            if not (public_path / 'test_cache' / nn_repo).exists():
                (public_path / 'test_cache' / nn_repo).mkdir()

            nn_manager.init_nnoperator(
                file_temp=public_path / 'test_cache' / temp_repo, file_dest=public_path / 'test_cache' / nn_repo
            )
            print(public_path / 'test_cache' / nn_repo / 'nn_operator.py')
            self.assertTrue((public_path / 'test_cache' / nn_repo / 'nn_operator.py').is_file())

    def test_init_pyoperator(self):
        with TemporaryDirectory(dir='./') as root:
            public_path = Path(root)
            temp_repo = 'pyoperator-template'
            git = GitUtils(author='towhee', repo=temp_repo)
            git.clone(local_repo_path=public_path / 'test_cache' / temp_repo)

            py_repo = 'py-operator'
            py_manager = OperatorManager('towhee', py_repo)
            if not (public_path / 'test_cache' / py_repo).exists():
                (public_path / 'test_cache' / py_repo).mkdir()

            py_manager.init_pyoperator(file_temp=public_path / 'test_cache' / temp_repo, file_dest=public_path / 'test_cache' / py_repo)
            self.assertTrue((public_path / 'test_cache' / py_repo / 'py_operator.py').is_file())


if __name__ == '__main__':
    unittest.main()
