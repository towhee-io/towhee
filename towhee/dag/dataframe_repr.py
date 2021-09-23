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
import logging

from towhee.dag.base_repr import BaseRepr
from towhee.dag.variable_repr import VariableRepr


class DataframeRepr(BaseRepr):
    """`DataframeRepr` represents a single dataframe within a graph.

    A single dataframe is composed of multiple individual variables, each of which is
    required in the next operator stage within the graph.
    """
    def __init__(self):
        super().__init__()
        # `_columns` must be filled in after this representation is instantiated
        self._columns = []

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value: List[VariableRepr]):
        self._columns = value

    @staticmethod
    def is_valid(info: List[dict]) -> bool:
        """Check if the src is a valid YAML file to describe the dataframe(s) in Towhee.

        Args:
            info(`list`):
                The list loaded from the source file.
        """
        essentials = {'name', 'columns'}

        if not isinstance(info, list):
            logging.error('src is not a valid YAML file.')
            return False

        for i in info:
            if not isinstance(i, dict):
                logging.error('src is not a valid YAML file.')
                return False
            if not essentials.issubset(set(i.keys())):
                logging.error('src cannot descirbe the dataframe(s) in Towhee.')
                return False

        return True

    @staticmethod
    def from_yaml(file_or_src: str):
        """Import a YAML file decribing the dataframe(s).

        Args:
            file_or_src(`str`):
                YAML file (could be pre-loaded as string) to import.

        Returns:
            The dataframes we described in `file_or_src`.
        """
        dataframes = DataframeRepr.load_src(file_or_src)
        if not DataframeRepr.is_valid(dataframes):
            raise ValueError('file or src is not a valid YAML file to describe the dataframe(s) in Towhee.')

        res = {}
        all_df = set()

        for df in dataframes:
            dataframe = DataframeRepr()
            dataframe.name = df['name']
            all_df.add(dataframe.name)
            for col in df['columns']:
                dataframe.columns.append(VariableRepr(col['vtype'], col['dtype']))
            res[dataframe.name] = dataframe

        return res
