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
from typing import Dict

from towhee.runtime.check_utils import check_config


class NodeConfig:
    """
    The config of nodes.
    """
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @staticmethod
    def from_dict(src: Dict[str, str]):
        essentials = {'name'}
        check_config(src, essentials)

        return NodeConfig(src['name'])
