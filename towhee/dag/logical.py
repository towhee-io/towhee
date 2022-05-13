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


class TransformationMeta:
    """
    `TransformationMeta` records the meta data of each transfromation.

    Args:
        op (`Callable`):
            The operator.
        inputs (`List[towhee.DataCollection]`):
            The operator's input DataCollections.
        output (`towhee.DataCollection`):
    """

    def __init__(self, op, inputs, output):
        self.op = op
        self.inputs = inputs
        self.output = output
        self.is_active = False


class TransformationsDAG:
    """
    The DAG consists of all the transfromations.
    """

    def __init__(self):
        self._transformations = []
        self._output2t_map = {}

    def add_transformation(self, t: TransformationMeta):
        """
        Add a transformation to DAG.

        Args:
            t (`TransformationMeta`):
                The new transformation need to be added.
        """
        self._transformations.append(t)
        self._output2t_map[t.output] = t

    @property
    def transformations(self) -> List[TransformationMeta]:
        """
        Getter of all the transformations.
        """
        return self._transformations

    def find_parents(self, t: TransformationMeta) -> List[TransformationMeta]:
        """
        Find all the parents of a transformation by its inputs. If the parent corresponding
        to an input is not found, the returned value at that parent position will be None.

        Args:
            t (`TransformationMeta`):
                The transformation.

        Return (`list[TransformationMeta]`):
            The parents of the given transformation.
        """
        parents = []
        for i in t.inputs:
            if i in self._output2t_map:
                parents.append(self._output2t_map[i])
            else:
                parents.append(None)
        return parents
