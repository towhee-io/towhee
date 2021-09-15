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

import yaml
import requests
import os
from typing import List

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
    def is_format(info: list):
        """Check if the src is a valid YAML file to describe the dataframe(s) in Towhee.

        Args:
            info(`list`):
                The list loaded from the source file.
        """
        essentials = {'name', 'columns'}
        if not isinstance(info, list):
            raise ValueError('src is not a valid YAML file')
        for i in info:
            if not isinstance(i, dict):
                raise ValueError('src is not a valid YAML file')
            if not essentials.issubset(set(i.keys())):
                print(i)
                raise ValueError('src cannot descirbe the dataframe(s) in Towhee')

    @staticmethod
    def load_file(file: str) -> dict:
        """Load the dataframe(s) information from a local YAML file.

        Args:
            file:
                The file path.

        Returns:
            The list loaded from the YAML file that contains the dataframe(s) information.
        """
        with open(file, 'r', encoding='utf-8') as f:
            res = yaml.safe_load(f)
        if isinstance(res, dict):
            res = [res]
        DataframeRepr.is_format(res)
        return res

    @staticmethod
    def load_url(url: str) -> dict:
        """Load the dataframe(s) information from a remote YAML file.

        Args:
            file:
                The url points to the remote YAML file.

        Returns:
            The list loaded from the YAML file that contains the dataframe(s) information.
        """
        src = requests.get(url).text
        res = yaml.safe_load(src)
        if isinstance(res, dict):
            res = [res]
        DataframeRepr.is_format(res)
        return res

    @staticmethod
    def load_str(string: str) -> list:
        """Load the dataframe(s) information from a YAML file (pre-loaded as string).

        Args:
            string:
                The string pre-loaded from a YAML.

        Returns:
            The list loaded from the YAML file that contains the dataframe(s) information.
        """
        res = yaml.safe_load(string)
        if isinstance(res, dict):
            res = [res]
        DataframeRepr.is_format(res)
        return res

    @staticmethod
    def load_src(file_or_src: str) -> str:
        """Load the information for the representation.

        We support file from local file/HTTP/HDFS.

        Args:
            file_or_src(`str`):
                The source YAML file or the URL points to the source file or a str
                loaded from source file.

        returns:
            The YAML file loaded as list.
        """
        # If `file_or_src` is a loacl file
        if os.path.isfile(file_or_src):
            return DataframeRepr.load_file(file_or_src)
        # If `file_or_src` from HTTP
        elif file_or_src.lower().startswith('http'):
            return DataframeRepr.load_url(file_or_src)
        # If `file_or_src` is neither a file nor url
        return DataframeRepr.load_str(file_or_src)

    @staticmethod
    def from_yaml(file_or_src: str):
        """Import a YAML file decribing this dataframe.

        Args:
            file_or_src(`str`):
                YAML file (could be pre-loaded as string) to import.

        Returns:
            The dataframes we described in `file_or_src`
        """
        dataframes = DataframeRepr.load_src(file_or_src)

        res = {}

        for df in dataframes:
            dataframe = DataframeRepr()
            dataframe.name = df['name']
            for col in df['columns']:
                dataframe.columns.append(VariableRepr(col['vtype'], col['dtype']))
            res[dataframe.name] = dataframe

        return res
