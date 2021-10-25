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
from typing import List, Dict, Any

from towhee.dag.base_repr import BaseRepr
from towhee.dag.variable_repr import VariableRepr


class DataFrameRepr(BaseRepr):
    """
    A `DataFrameRepr` represents a single dataframe within a graph.

    A single dataframe is composed of multiple individual variables, each of which is
    required in the next operator stage within the graph.

    Args:
        name (`str`):
            The representation name.
        columns (`List[VariableRepr]`):
            The columns that the DataFrameRepr contains.
    """
    def __init__(self, name: str, columns: List[VariableRepr]):
        self._name = name
        self._columns = columns

    @property
    def name(self):
        return self._name

    @property
    def columns(self):
        return self._columns

    @staticmethod
    def from_dict(info: Dict[str, Any]) -> 'DataFrameRepr':
        """
        Generate a DataframeRepr from a description dict.

        Args:
            info (`Dict[str, Any]`):
                A dict to describe the Dataframe.

        Returns:
            (`towhee.dag.DataFrameRepr`)
                The DataFrameRepr object.
        """
        if not DataFrameRepr.is_valid(info, {'name', 'columns'}):
            raise ValueError('Invalid dataframe info.')
        var_reprs = []
        df_name = info['name']
        for index, col in enumerate(info['columns']):
            var_reprs.append(VariableRepr(col['name'], col['vtype'], f'{df_name}_col_{index}'))
        return DataFrameRepr(df_name, var_reprs)
