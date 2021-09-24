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
from typing import List, Dict

from towhee.dag.base_repr import BaseRepr
from towhee.dag.variable_repr import VariableRepr


class DataframeRepr(BaseRepr):
    """`DataframeRepr` represents a single dataframe within a graph.

    A single dataframe is composed of multiple individual variables, each of which is
    required in the next operator stage within the graph.

    Args:
        name:
            The representation name.
        src:
            The information of this dataframe.
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
    def from_dict(info: Dict[str, any]) -> 'DataframeRepr':
        """
        info:
            dataframe info
            {
                'name': 'example_df',
                'columns': [
                    {'vtype': 'int'},
                    {'vtype': 'float'},
                ],
            }
        Returns:
            DataframeRepr obj
        """
        if not DataframeRepr.is_valid(info, {'name', 'columns'}):
            raise ValueError(
                'Invalid dataframe info.'
            )
        var_reprs = []
        for c in info['columns']:
            var_reprs.append(VariableRepr(c['vtype']))
        return DataframeRepr(info['name'], var_reprs)
