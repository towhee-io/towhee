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
import shutil
import unittest
import argparse
import os
from pathlib import Path

from towhee.command.initialize import InitCommand

public_path = Path(__file__).parent.parent.resolve()


class TestCmdline(unittest.TestCase):
    """
    Unittests for towhee cmdline.
    """
    def test_init(self):
        pyrepo = 'towhee/init-pyoperator'
        nnrepo = 'towhee/init-nnoperator'
        repo_path = public_path / 'mock_operators'
        os.chdir(str(repo_path))
        args_init_pyop = argparse.Namespace(action='init', type='pyop', dir=str(repo_path / 'init-pyoperator'), uri=pyrepo, local=True)
        args_init_nnop = argparse.Namespace(action='init', type='nnop', dir=str(repo_path / 'init-nnoperator'), uri=nnrepo, local=True)

        InitCommand(args_init_pyop)()
        InitCommand(args_init_nnop)()
        self.assertTrue((repo_path / 'init-pyoperator' / 'init_pyoperator.py').is_file())
        self.assertTrue((repo_path / 'init-nnoperator' / 'init_nnoperator.py').is_file())

        shutil.rmtree(str(repo_path / 'init-pyoperator'))
        shutil.rmtree(str(repo_path / 'init-nnoperator'))


if __name__ == '__main__':
    unittest.main()
