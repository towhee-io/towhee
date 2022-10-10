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

from typing import Dict, Any

from towhee.runtime.node_repr import NodeRepr
from towhee.runtime.schema_repr import SchemaRepr


class DAGRepr:
    """
    A `DAGRepr` represents a complete DAG.

    Args:
        dag_type (`str`): The type of DAG type, such as 'local', 'triton' and 'mix', defaults to 'local'.
        nodes (`Dict[str, NodeRepr]`): All nodes in the dag.
        schemas (`Dict[str, SchemaRepr]`): All schemas in the dag for each node.
    """
    def __init__(self, nodes: Dict[str, NodeRepr], schemas: Dict[str, SchemaRepr], dag_type: str = 'local'):
        self._nodes = nodes
        self._schemas = schemas
        self._dag_type = dag_type

    @property
    def dag_type(self) -> str:
        return self._dag_type

    @property
    def nodes(self) -> Dict:
        return self._nodes

    @property
    def schemas(self) -> Dict:
        return self._schemas

    @staticmethod
    def from_dict(dag: Dict[str, Any], dag_type: str = 'local'):
        nodes = dict((name, NodeRepr.from_dict(name, dag[name])) for name in dag)
        schemas = dict((name, SchemaRepr(name, dag[name]['inputs'], dag[name]['outputs'])) for name in dag)
        return DAGRepr(nodes, schemas, dag_type)
