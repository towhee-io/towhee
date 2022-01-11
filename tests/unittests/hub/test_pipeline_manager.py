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

from towhee.hub.pipeline_manager import PipelineManager

public_path = Path(__file__).parent.parent.resolve()


@unittest.skip('Not pass')
class TestPipelineManager(unittest.TestCase):
    """
    Unittest for OperatorManager.
    """
    def test_init(self):
        file_src = public_path / 'mock_pipelines' / 'pipeline_template'
        file_dst = public_path / 'test_cache' / 'pipelines' / 'towhee' / 'ci_test' / 'main'

        if file_dst.is_dir():
            rmtree(file_dst)

        rm = PipelineManager('towhee', 'ci-test')

        rm.init(file_src, file_dst)

        self.assertIn('ci_test.yaml', os.listdir(file_dst))

        rmtree(file_dst.parent)
