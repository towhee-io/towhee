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

from towhee.engine.engine import Engine, EngineConfig
from towhee.hub.file_manager import FileManagerConfig, FileManager


UNITTESTS_ROOT = os.path.dirname(os.path.abspath(__file__))

VIT_DIR = os.path.join(UNITTESTS_ROOT, 'models/vit')
MODEL_RESOURCE = os.path.join(UNITTESTS_ROOT, 'models/resource')


CACHE_PATH = Path(__file__).parent.resolve()

new_cache = (CACHE_PATH/'test_cache')
pipeline_cache = (CACHE_PATH/'test_util')
operator_cache = (CACHE_PATH/'mock_operators')
fmc = FileManagerConfig()
fmc.update_default_cache(new_cache)
pipelines = list(pipeline_cache.rglob('*.yaml'))
operators = [f for f in operator_cache.iterdir() if f.is_dir()]
fmc.cache_local_pipeline(pipelines)
fmc.cache_local_operator(operators)
FileManager(fmc)

conf = EngineConfig()
conf.cache_path = CACHE_PATH
engine = Engine()
if not engine.is_alive():
    engine.start()
