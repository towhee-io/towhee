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
from requests.exceptions import HTTPError

from towhee.utils.git_utils import GitUtils
from towhee.hub.pipeline_manager import PipelineManager

public_path = Path(__file__).parent.parent.resolve()


class TestPipelineManager(unittest.TestCase):
    """
    Unit test for PipelineManager.
    """
    def test_create(self):
        repo = 'pipeline-operator'
        manager = PipelineManager('towhee', repo)
        with self.assertRaises(HTTPError):
            manager.create('psw')

    def test_exists_create(self):
        repo = 'ci-test'
        manager = PipelineManager('towhee', repo)
        manager.create('psw')

    def test_init_pipeline(self):
        temp_repo = 'pipeline-template'
        git = GitUtils(author='towhee', repo=temp_repo)
        git.clone(local_repo_path=public_path / 'test_cache' / temp_repo)

        pipeline_repo = 'init-pipeline'
        pipeline_manager = PipelineManager('towhee', pipeline_repo)
        if not (public_path / 'test_cache' / pipeline_repo).exists():
            (public_path / 'test_cache' / pipeline_repo).mkdir()

        pipeline_manager.init_pipeline(file_temp=public_path / 'test_cache' / temp_repo, file_dest=public_path / 'test_cache' / pipeline_repo)
        self.assertTrue(pipeline_manager.check(public_path / 'test_cache' / pipeline_repo))
        rmtree(public_path / 'test_cache' / temp_repo)
        rmtree(public_path / 'test_cache' / pipeline_repo)

    def test_check(self):
        repo = 'ci-test'
        manager = PipelineManager('towhee', repo)
        self.assertTrue(manager.check(public_path / 'mock_pipelines/ci_test'))


if __name__ == '__main__':
    unittest.main()
