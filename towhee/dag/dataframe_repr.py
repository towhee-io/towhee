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


from collections import OrderedDict
from towhee.dataframe.dataframe import DFIterator
from towhee.dag.variable_repr import VariableRepr


class DFRepr:
    """
    A collection of VariableRepr
    """
    def __init__(self, name = None, vars = None):
        """
        Args:
            name: name of the DFRepr
            vars: a list of VariableRepr
        """
        self.name = name
        self.columns =  OrderedDict((var.name, var) for var in vars)
        self._iter = DFIterator(df = None)

    def __getattr__(self, key: str):
        """
        Access a VariableRepr as DFRepr's attribute
        Args:
            key: the name of the VariableRepr
        """
        return self.columns[key]

    def __setitem__(self, key: str, value: VariableRepr):
        """
        Set a VariableRepr
        Args:
            key: the name of the VariableRepr
            value: the object of the VariableRepr
        """
        self.columns[key] = value
    
    def __getitem__(self, key):
        """
        Get a VariableRepr by its name
        Args:
            key: the name of the VariableRepr
        """
        return self.columns[key]
    
    def from_input_annotations(self, func: function):
        """
        Parse variables from a function's input annotations
        """
        raise NotImplementedError

    def from_output_annotations(self, func: function):
        """
        Parse variables from a function's output annotations
        """
        raise NotImplementedError