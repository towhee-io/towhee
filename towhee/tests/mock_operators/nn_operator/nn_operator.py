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

from towhee.operator import NNOperator
from towhee.utils.log import engine_log


class TestNNOperator(NNOperator):
    """
    A test NNOperator with no functionality.
    """
    def __init__(self, framework: str = 'pytorch'):
        super().__init__()
        self._framework = framework

    @property
    def framework(self):
        return self._framework

    @framework.setter
    def framework(self, framework: str):
        self._framework = framework

    def __call__(self):
        engine_log.info('I\'m an NNOperator.')
