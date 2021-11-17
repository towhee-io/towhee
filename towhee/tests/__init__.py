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


from pathlib import Path
# from towhee.engine.engine_v2 import Engine, EngineConfig as v2_Engine, v2_EngineConfig
# from towhee.engine.engine import Engine, EngineConfig as v1_Engine, v1_EngineConfig

import towhee.engine.engine_v2 as engine_v2
import towhee.engine.engine as engine

CACHE_PATH = Path(__file__).parent.resolve()

v2_conf = engine_v2.EngineConfig()
v2_conf.cache_path = CACHE_PATH
v2_conf.sched_interval_ms = 20
v2_engine = engine_v2.Engine()
if not v2_engine.is_alive():
    v2_engine.start()


v1_conf = engine.EngineConfig()
v1_conf.cache_path = CACHE_PATH
v1_conf.sched_interval_ms = 20
v1_engine = engine.Engine()
if not v1_engine.is_alive():
    v1_engine.start()
