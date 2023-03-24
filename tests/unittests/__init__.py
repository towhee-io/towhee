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


UNITTESTS_ROOT = os.path.dirname(os.path.abspath(__file__))

VIT_DIR = os.path.join(UNITTESTS_ROOT, 'models/vit')
MODEL_RESOURCE = os.path.join(UNITTESTS_ROOT, 'models/resource')


CACHE_PATH = Path(__file__).parent.resolve()

operator_cache = (CACHE_PATH/'mock_operators')
pipeline_cache = (CACHE_PATH/'mock_pipelines')
# dc2 op cache
os.environ['TEST_OP_CACHE'] = str(operator_cache)
os.environ['TEST_PIPE_CACHE'] = str(pipeline_cache)
