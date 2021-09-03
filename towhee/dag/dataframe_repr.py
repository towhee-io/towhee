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


from collections import OrderedDict

from towhee.base_repr import BaseRepr
from towhee.dataframe.dataframe import DFIterator
from towhee.dag.variable_repr import VariableRepr


class DataframeRepr(BaseRepr):
    """`DataframeRepr` represents a single dataframe within a graph. A single dataframe
    is composed of multiple individual variables, each of which is required in the next
    operator stage within the graph.

    Args:
        name:
            The representation name.
        vars:
            A list of variables that can be accessed from within the dataframe.
        from_op:
            `OperatorRepr` instance which represents the source operator.
        to_op:
            `OperatorRepr` instance which represents the destination operator.
    """

    def __init__(self, name: str, vars: list[VariableRepr]):
        self._name = name

    def __getitem__(self, key) -> VariableRepr:
        """Access a variable representation via dictionary-like indexing.

        Args:
            key:
                Name of the variable representation to access.

        Returns:
            The variable corresponding to the specified key, or None if an invalid key
            was provided.
        """
        return self._columns.get(key)

    def __setitem__(self, key: str, value: VariableRepr):
        """Sets a single variable representation within the dataframe.

        Args:
            key:
                Variable name.
            value:
                A pre-instantiated `VariableRepr` instance.
        """
        self._columns[key] = value

    def from_input_annotations(self, func: function):
        """Parse variables from an operator's input annotations.

        Args:
            func:
        """
        for (name, kind) in func.__annotations__.items():
            if kind == Variable:
                # Ignore return types in annotation dictionary.
                if name == 'return':
                    continue
                # Create a valid variable representation based on each parameter
                # name and type.
                var_rep = VariableRepr()

    def from_output_annotations(self, func: function):
        """Parse variables from an operator's output annotations.
        """
        if func.__annotations__.get('return') == 'Variable':
