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

from towhee.command.develop import SetupCommand, UninstallCommand
from towhee.command.repo import RepoCommand
# from towhee.command.execute import ExecuteCommand

public_path = Path(__file__).parent.parent.resolve()


class TestCmdline(unittest.TestCase):
    """
    Unittests for towhee cmdline.
    """
    def test_develop(self):
        repo = 'add_operator'
        repo_path = public_path / 'mock_operators' / repo
        os.chdir(str(repo_path))
        args_dev = argparse.Namespace(action='install', namespace='test', path=str(repo_path), develop=True, user=True)
        args_ins = argparse.Namespace(action='install', namespace='test', path=str(repo_path), develop=False, user=True)
        args_unins = argparse.Namespace(action='uninstall', namespace='test', path=str(repo_path))

        SetupCommand(args_ins)()
        UninstallCommand(args_unins)()
        SetupCommand(args_dev)()
        UninstallCommand(args_unins)()

    def test_create(self):
        pyrepo = 'create_pyoperator'
        nnrepo = 'cmd/create_nnoperator'
        repo_path = public_path / 'mock_operators'
        os.chdir(str(repo_path))
        args_create_pyop = argparse.Namespace(action='create', type='pyop', dir=str(repo_path), uri=pyrepo, local=True)
        args_create_nnop = argparse.Namespace(action='create', type='nnop', framework='tensorflow', dir=str(repo_path), uri=nnrepo, local=True)

        RepoCommand(args_create_pyop)()
        RepoCommand(args_create_nnop)()
        self.assertTrue((repo_path / pyrepo / 'create_pyoperator.py').is_file())
        self.assertTrue((repo_path / 'create_nnoperator' / 'create_nnoperator.py').is_file())
        self.assertTrue((repo_path / 'create_nnoperator' / 'tensorflow').is_dir())

        shutil.rmtree(str(repo_path / pyrepo))
        shutil.rmtree(str(repo_path / 'create_nnoperator'))

    # TODO: enable this test case with pytest @shiyu22
    # def test_run(self):
    #     img_path = 'https://github.com/towhee-io/towhee/blob/main/towhee_logo.png?raw=true'
    #     args_1 = argparse.Namespace(command='run', input=img_path, output=public_path / 'test_cache', pipeline='towhee/image-embedding-resnet50')
    #     ExecuteCommand(args_1)()
    #     self.assertTrue((public_path / 'test_cache/towhee_output.txt').is_file())

if __name__ == '__main__':
    unittest.main()
