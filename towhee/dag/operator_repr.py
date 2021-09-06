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


from towhee.base_repr import BaseRepr
from towhee.dag.dataframe_repr import DataframeRepr
#from towhee.dag.variable_repr import VariableRepr
#from towhee.dag.variable_repr import VariableReprSet


class OperatorRepr(BaseRepr):
    """This class encapsulates operator representations at compile-time.

    Args:
        name:
            Name of the operator represented by this object.
        df_in:
            Input dataframes(s) to this object.
        df_out:
            This operator's output dataframe.
    """

    def __init__(self, name: str, df_in: DataframeRepr, df_out: DataframeRepr):
        super().__init__(name)
        self._df_in = df_in
        self._df_out = df_out
        self._iter_in = None
        self._iter_out = None

    @property
    def df_in(self):
        return self._df_in

    @property
    def df_out(self):
        return self._df_out

    @df_in.setter
    def df_in(self, value: DataframeRepr):
        self._df_in = value

    @df_out.setter
    def df_out(self, value: DataframeRepr):
        self._df_out = value
