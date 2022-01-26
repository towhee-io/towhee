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
import os
import subprocess
from shutil import rmtree
from pathlib import Path

from towhee.utils.git_utils import GitUtils

public_path = Path(__file__).parent.parent.resolve()
repo_path = public_path / 'test_cache/pipelines/towhee/ci_test'
print(repo_path)
git = GitUtils(author='towhee', repo='ci-test')
if repo_path.is_dir():
    rmtree(repo_path)
git.clone(local_repo_path=repo_path)


class TestGitUtils(unittest.TestCase):
    """
    Unit test for git utils.
    """
    def test_clone(self):
        self.assertTrue(repo_path.is_dir())
        self.assertTrue((repo_path / 'ci_test.yaml').is_file())

    def test_add(self):
        cwd = Path.cwd()
        os.chdir(repo_path)
        res_code = git.add()
        self.assertEqual(res_code, 0)
        os.chdir(cwd)

    def test_pull(self):
        cwd = Path.cwd()
        os.chdir(repo_path)
        subprocess.check_call(['git', 'config', 'pull.rebase', 'true'])
        res_code = git.pull()
        self.assertEqual(res_code, 0)
        os.chdir(cwd)

    def test_status(self):
        cwd = Path.cwd()
        os.chdir(repo_path)
        res = git.status()
        self.assertIn('up to date with', res)
        os.chdir(cwd)
