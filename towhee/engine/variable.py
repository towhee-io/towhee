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


from towhee.dataframe.dataframe import DFIterator, DataFrame
from towhee.engine.operator_context import OperatorContext


class Variable:
    """
    A Variable can be part of an Operator's inputs or outputs.
    """

    def __init__(self, name: str, df: DataFrame, df_iter: DFIterator, op_ctx: OperatorContext):
        """
        Args:
            name: the Variable's name
            df: the DataFrame this Variable belongs to
            iter: the DataFrame's iterator
            op_ctx: the OperatorContext this Variable belongs to.
        """
        self.name = name
        self.df = df
        self.df_iter = df_iter
        self.op_ctx = op_ctx
