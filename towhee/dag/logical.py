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


class OpCallMeta:
    """
    `OpCallMeta` records the meta data of each operator call.

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


class LogicalDAG:
    """
    The DAG consists of all the transfromations.
    """

    def __init__(self):
        self._op_calls = []

    def add_op_call(self, op_call: OpCallMeta):
        """
        Add a operator call meta to DAG.

        Args:
            op_call (`OpCallMeta`):
                The new operator call meta need to be added.
        """
        self._op_calls.append(op_call)

    @property
    def op_calls(self) -> List[OpCallMeta]:
        """
        Getter of all the operator call metas.
        """
        return self._op_calls
