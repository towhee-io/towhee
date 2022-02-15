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


from typing import Optional


class Variable:
    """A `Variable` is a blob of data which can be one of many data types such as:
    `bool` (scalar), `int` (scalar), `float` (scalar), `tuple` (array), `list` (array),
    `np.ndarray` (array), `string` (misc).

    Args:
        name:
            Variable name; should be identical to its representation counterpart.

        vtype:
            type name: int, float, str, bytes ...
        value:
            variable data

    """

    def __init__(self, vtype: str, value: any, name: Optional[str] = None):
        self._vtype = vtype
        self._value = value
        self._name = name
    
    def __str__(self):
        return str(self._value)

    @property
    def value(self) -> any:
        return self._value

    @property
    def vtype(self) -> str:
        return self._vtype
