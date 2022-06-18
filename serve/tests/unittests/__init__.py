# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from pathlib import Path

from towhee.hub.file_manager import FileManagerConfig, FileManager


UNITTESTS_ROOT = os.path.dirname(os.path.abspath(__file__))

CACHE_PATH = Path(__file__).parent.parent.parent.parent / 'tests' / 'unittests'

new_cache = (CACHE_PATH / 'test_cache')
operator_cache = (CACHE_PATH / 'mock_operators')
print(operator_cache, '---------')
fmc = FileManagerConfig()
fmc.update_default_cache(new_cache)
operators = [f for f in operator_cache.iterdir() if f.is_dir()]
print(operators)
fmc.cache_local_operator(operators)
FileManager(fmc)
