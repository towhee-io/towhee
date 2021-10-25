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

from towhee.dag.base_repr import BaseRepr


class VariableRepr(BaseRepr):
    """
    The representation of a variable at compile-phase.

    Args:
        name (`str`):
            The representation name.
        vtype (`str`):
            This can be one of many possible variable types, such as a numpy array or
            PyTorch tensor.
        dtype (`str`):
            A string or instance of `numpy.dtype` indicating the internal data type for
            this variable.
    """
    def __init__(self, name: str, vtype: str, identifier: str = None):
        super().__init__(name)
        self._vtype = vtype
        self._id = identifier

    @property
    def vtype(self):
        return self._vtype

    # @property
    # def identifier(self):
    #     return self._id
