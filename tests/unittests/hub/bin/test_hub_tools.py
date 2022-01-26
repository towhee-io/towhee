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

from towhee.hub.bin.hub_tools import get_repo_obj
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
