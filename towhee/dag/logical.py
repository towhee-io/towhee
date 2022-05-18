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

from typing import List


class DCDependency:
    """
    `OpCallMeta` records the meta data of each operator call.

    Args:
        op (`Callable`):
            The operator.
        inputs (`List[towhee.DataCollection]`):
            The operator's input DataCollections.
        output (`towhee.DataCollection`):
            The operator's output DataCollection.
    """

    def __init__(self, op, inputs, output):
        self.op = op
        self.inputs = inputs
        self.output = output


class DCDependencyGraph:
    """
    The DAG consists of all the DataCollection dependencies.
    """

    def __init__(self):
        self._dc_dependencies = []

    def add_dc_dependency(self, dependency: DCDependency):
        """
        Add a DataCollection dependency to DAG.

        Args:
            dependency (`DCDependency`):
                The DataCollection dependency need to be added.
        """
        self._dc_dependencies.append(dependency)

    @property
    def dc_dependencies(self) -> List[DCDependency]:
        """
        Getter of all the DataCollection dependencies.
        """
        return self._dc_dependencies
