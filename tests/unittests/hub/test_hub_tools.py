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
import argparse
from pathlib import Path
from shutil import rmtree

from towhee.hub.bin.hub_tools import get_repo_obj, init_repo
from towhee.hub.repo_manager import RepoManager
from towhee.hub.operator_manager import OperatorManager
from towhee.hub.pipeline_manager import PipelineManager
from towhee.utils.git_utils import GitUtils


class TestHubTools(unittest.TestCase):
    """
    Unittests for hub tools.
    """
    def test_get_repo_obj(self):
        args_1 = argparse.Namespace(command='download', author='towhee', repo='ci-test')
        args_2 = argparse.Namespace(command='clone', author='towhee', repo='ci-test')
        args_3 = argparse.Namespace(command='generate-yaml', author='towhee', repo='ci-test-operator')
        args_4 = argparse.Namespace(command='', classes='pipeline', author='towhee', repo='ci-test')
        args_5 = argparse.Namespace(command='', classes='operator', author='towhee', repo='ci-test')

        repo_obj_1 = get_repo_obj(args_1)
        repo_obj_2 = get_repo_obj(args_2)
        repo_obj_3 = get_repo_obj(args_3)
        repo_obj_4 = get_repo_obj(args_4)
        repo_obj_5 = get_repo_obj(args_5)

        self.assertIsInstance(repo_obj_1, RepoManager)
        self.assertIsInstance(repo_obj_2, GitUtils)
        self.assertIsInstance(repo_obj_3, OperatorManager)
        self.assertIsInstance(repo_obj_4, PipelineManager)
        self.assertIsInstance(repo_obj_5, OperatorManager)

    def test_init_repo(self):
        public_path = Path(__file__).parent.parent.resolve()
        pipeline_dir = public_path / 'test_cache' / 'pipelines' / 'temp'
        nnop_dir = public_path / 'test_cache' / 'operators' / 'nntemp'
        pyop_dir = public_path / 'test_cache' / 'operators' / 'pytemp'
        args_pipeline = argparse.Namespace(command='init', author='towhee', repo='ci-test', classes='pipeline', dir=pipeline_dir)
        args_nnop = argparse.Namespace(
            command='init', author='towhee', repo='ci-test-operator', classes='nnoperator', framework='pytorch', dir=nnop_dir
        )
        args_pyop = argparse.Namespace(command='init', author='towhee', repo='ci-test-operator', classes='pyoperator', dir=pyop_dir)

        init_repo(args_pipeline)
        init_repo(args_nnop)
        init_repo(args_pyop)

        self.assertTrue(nnop_dir.is_dir())
        self.assertTrue(pyop_dir.is_dir())
        self.assertTrue(pipeline_dir.is_dir())

        self.assertTrue((pipeline_dir / 'ci_test' / 'ci_test.yaml').is_file())
        self.assertTrue((nnop_dir / 'ci_test_operator' / 'ci_test_operator.py').is_file())
        self.assertTrue((nnop_dir / 'ci_test_operator' / 'pytorch').is_dir())
        self.assertTrue((pyop_dir / 'ci_test_operator' / 'ci_test_operator.py').is_file())

        rmtree(pipeline_dir)
        rmtree(nnop_dir)
        rmtree(pyop_dir)
