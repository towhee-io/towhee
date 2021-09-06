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


from towhee.dataframe.dataframe import DataFrame
from towhee.engine.operator_context import OperatorContext


class Variable:
    """A `Variable` is a blob of data which can be one of many data types such as:
    `bool` (scalar), `int` (scalar), `float` (scalar), `tuple` (array), `list` (array),
    `np.ndarray` (array), `string` (misc).

    Args:
        name:
            Variable name; should be identical to its representation counterpart.
        df:
            The DataFrame this Variable belongs to.
        op_ctx:
            The OperatorContext this Variable belongs to.
    """

    def __init__(self, name: str, df: DataFrame, op_ctx: OperatorContext):
        self._name = name
        self._df = df
        self._op_ctx = op_ctx
