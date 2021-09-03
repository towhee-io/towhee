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


from operator_repr import OperatorRepr


class GraphRepr(BaseRepr):
    """This class presents a complete representation of a graph and its individual
    subcomponents, including Operators, Dataframes, and Variables. Graph representations
    are used during execution to load functions and pass data to the correct operators.
    """

    def __init__(self, builder_id: str):
        self.builder_id = builder_id
        self._op_dict = []
        self._df_dict = []

    def op(self):
        return self._op_dict

    def from_yaml(self, yaml: str):
        """Import a YAML file describing this graph.

        Args:
            yaml:
                YAML file (pre-loaded as string) to import.
        """
        raise NotImplementedError

    def to_yaml(self):
        """Export a YAML file describing this graph.

        Returns:
            A string with the graph's serialized contents.
        """
        raise NotImplementedError
