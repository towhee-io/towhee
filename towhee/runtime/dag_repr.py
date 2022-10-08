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

from typing import Dict, List
from towhee.runtime.node_repr import NodeRepr


class DAGRepr:
    def __init__(self, dag: dict):
        self._dag = dag

    @property
    def dag(self) -> Dict:
        return self._dag

    def get_nodes(self) -> List:
        nodes_list = []
        for name in self._dag:
            nodes_list.append(NodeRepr.from_dict(name, self._dag[name]))
        return nodes_list
