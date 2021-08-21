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


class VariableRepr:
    """
    The representation of a variable at compile-phase
    """

    def __init__(self, name: str, from_op: str = None):
        self.from_op = from_op
        self.to_op = []
        self.name = name
        self.type = None
        self.dtype = None


class VariableSet:
    """
    A set of VariableRepr
    """
    def __init__(self, value):
        if isinstance(value, list):
            self.from_list(value)
        elif isinstance(value, dict):
            self._var_dict = value

    def __getattr__(self, name: str):
        raise NotImplementedError
        return self._var_dict[name]

    def from_dict(self, var_dict: dict):
        """
        Add variables from dict
        """
        raise NotImplementedError
    
    def from_list(self, vars: list):
        """
        Add variables from list
        """
        raise NotImplementedError

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

    def add_var(self, key: str, var: VariableRepr):
        """
        Add a variable
        """
        raise NotImplementedError

