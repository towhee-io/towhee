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
from .operator_registry import OperatorRegistry # pylint: disable=import-outside-toplevel


register = OperatorRegistry.register
resolve = OperatorRegistry.resolve


DEFAULT_LOCAL_CACHE_ROOT = Path.home() / '.towhee'
LOCAL_PIPELINE_CACHE = DEFAULT_LOCAL_CACHE_ROOT / 'pipelines'
LOCAL_OPERATOR_CACHE = DEFAULT_LOCAL_CACHE_ROOT / 'operators'

MOCK_PIPELINES = str(Path(__file__).parent.parent /
                     'tests/test_util/resnet50_embedding.yaml')

MOCK_OPS = str(Path(__file__).parent.parent / 'tests/mock_operators')
MOCK_PIPES = str(Path(__file__).parent.parent / 'tests/mock_pipelines')


