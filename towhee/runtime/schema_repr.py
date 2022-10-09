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

from typing import Union, Tuple


class SchemaRepr:
    """
    A `SchemaRepr` represents the node schema.

    Args:
        name (`str`): The name of the node.
        inputs (`Union[str, Tuple]`): The inputs schema of the node.
        outputs (`Union[str, Tuple]`): The outputs schema of the node.
    """
    def __init__(self, name: str, inputs: Union[str, Tuple], outputs: Union[str, Tuple]):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self) -> Union[str, Tuple]:
        return self._inputs

    @property
    def outputs(self) -> Union[str, Tuple]:
        return self._outputs
