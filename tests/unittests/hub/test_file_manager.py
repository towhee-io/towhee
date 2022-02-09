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

from towhee.hub.file_manager import FileManagerConfig

fmc = FileManagerConfig()


class TestFileManager(unittest.TestCase):
    """
    Unit test for FileManager
    """
    def test_add_remove_cache_path(self):
        fmc.add_cache_path(Path('test_path'))
        self.assertEqual(fmc.cache_paths[0], Path('test_path'))
        fmc.remove_cache_path(Path('test_path'))
        self.assertNotIn(Path('test_path'), fmc.cache_paths)

    def test_reset_cache_path(self):
        fmc.reset_cache_path()
        self.assertEqual(len(fmc.cache_paths), 1)
