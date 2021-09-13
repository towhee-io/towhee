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

# builder = GraphBuilder()
# data1 = data0.map(op1)
# data2 = data1.batch(op2, bs=64)

import yaml

from towhee.base_repr import BaseRepr
from towhee.dag.variable_repr import VariableRepr


class DataframeRepr(BaseRepr):
    """`DataframeRepr` represents a single dataframe within a graph.

    A single dataframe is composed of multiple individual variables, each of which is
    required in the next operator stage within the graph.

    Args:
        name:
            The representation name.
        src:
            The information of this dataframe.
    """
    def __init__(self, name: str = None, src: str = None):
        super().__init__(name)
        # `_columns` must be filled in after this representation is instantiated
        self._columns = []
        if src:
            self.from_yaml(src)

    def __getitem__(self, index: int) -> VariableRepr:
        """Access a variable representation via dictionary-like indexing.

        Args:
            key:
                Name of the variable representation to access.

        Returns:
            The variable corresponding to the specified key, or None if an invalid key
            was provided.
        """
        return self._columns[index]

    def __setitem__(self, index: int, value: VariableRepr):
        """Sets a single variable representation within the dataframe.

        Args:
            key:
                Variable name.
            value:
                A pre-instantiated `VariableRepr` instance.
        """
        self._columns[index] = value

    def from_yaml(self, src: str):
        """Import a YAML file decribing this dataframe.

        Args:
            src:
                YAML file (pre-loaded as string) to import.
        """
        df = yaml.safe_load(src)
        self._name = df['name']
        for col in df['columns']:
            self._columns.append(VariableRepr(col['vtype'], col['dtype']))
