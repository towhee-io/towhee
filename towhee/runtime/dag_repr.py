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

from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.schema_repr import SchemaRepr


class DAGRepr:
    """
    A `DAGRepr` represents a complete DAG.

    Args:
        dag (`Dict[str, Any]`): The DAG dictionary.
    """
    def __init__(self, dag: dict):
        self._dag = dag
        self._nodes = dict((name, NodeRepr.from_dict(name, self._dag[name])) for name in self._dag)
        self._schemas = dict((name, SchemaRepr(name, self._dag[name]['inputs'], self._dag[name]['outputs'])) for name in self._dag)

    @property
    def dag(self) -> Dict:
        return self._dag

    @property
    def nodes(self) -> Dict:
        return self._nodes

    @property
    def schemas(self) -> Dict:
        return self._schemas
