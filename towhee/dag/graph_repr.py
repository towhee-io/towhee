# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import os
import requests
from typing import Dict

from towhee.dag.base_repr import BaseRepr
from towhee.dag.dataframe_repr import DataframeRepr
from towhee.dag.operator_repr import OperatorRepr


class GraphRepr(BaseRepr):
    """This class presents a complete representation of a graph.

    A graph contains individual subcomponents, including Operators, Dataframes, and
    Variables. Graph representations are used during execution to load functions and
    pass data to the correct operators.

    Args:
        name(`str`):
            The representation name.
        file_or_url(`str`):
            The file or remote url that stores the information of this representation.
    """
    def __init__(self):
        super().__init__()
        self._operators = {}
        self._dataframes = {}

    @property
    def operators(self) -> Dict[str, OperatorRepr]:
        return self._operators

    @property
    def dataframes(self) -> Dict[str, DataframeRepr]:
        return self._dataframes

    @operators.setter
    def operators(self, value: Dict[str, DataframeRepr]):
        self._operators = value

    @dataframes.setter
    def dataframes(self, value: Dict[str, DataframeRepr]):
        self._dataframes = value

    @staticmethod
    def is_format(info: dict):
        """Check if the src is a valid YAML file to describe a DAG in Towhee.

        Args:
            info(`dict`):
                The dict loaded from the source file.
        """
        essentials = {'graph', 'operators', 'dataframes'}
        if not isinstance(info, dict):
            raise ValueError('src is not a valid YAML file')
        if not essentials.issubset(set(info.keys())):
            raise ValueError('src cannot descirbe a DAG in Towhee')

    @staticmethod
    def load_file(file: str) -> dict:
        """Load the graph information from a local YAML file.

        Args:
            file:
                The file path.

        Returns:
            The dict loaded from the YAML file that contains the graph information.
        """
        with open(file, 'r', encoding='utf-8') as f:
            res = yaml.safe_load(f)
        GraphRepr.is_format(res)
        return res

    @staticmethod
    def load_url(url: str) -> dict:
        """Load the graph information from a remote YAML file.

        Args:
            file:
                The url points to the remote YAML file.

        Returns:
            The dict loaded from the YAML file that contains the graph information.
        """
        src = requests.get(url).text
        res = yaml.safe_load(src)
        GraphRepr.is_format(res)
        return res

    @staticmethod
    def load_str(string: str) -> dict:
        """Load the graph information from a YAML file (pre-loaded as string).

        Args:
            string:
                The string pre-loaded from a YAML.

        Returns:
            The dict loaded from the YAML file that contains the graph information.
        """
        res = yaml.safe_load(string)
        GraphRepr.is_format(res)
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
            The YAML file loaded as dict.
        """
        # If `file_or_src` is a loacl file
        if os.path.isfile(file_or_src):
            return GraphRepr.load_file(file_or_src)
        # If `file_or_src` from HTTP
        elif file_or_src.lower().startswith('http'):
            return GraphRepr.load_url(file_or_src)
        # If `file_or_src` is neither a file nor url
        return GraphRepr.load_str(file_or_src)

    @staticmethod
    def from_yaml(file_or_src: str):
        """Import a YAML file describing this graph.

        Args:
            file_or_src(`str`):
                YAML file (could be pre-loaded as string) to import.

        Returns:
            The DAG we described in `file_or_src`.
        """
        graph = GraphRepr()
        graph_dict = graph.load_src(file_or_src)

        # load name
        graph.name = graph_dict['graph']['name']

        # load dataframes
        # for df in graph_dict['dataframes']:
        graph.dataframes = DataframeRepr.from_yaml(yaml.safe_dump(graph_dict['dataframes'], default_flow_style=False))

        # load operators
        # for op in graph_dict['operators']:
        graph.operators = OperatorRepr.from_yaml(yaml.safe_dump(graph_dict['operators'], default_flow_style=False), graph.dataframes)

        return graph

    def to_yaml(self) -> str:
        """Export a YAML file describing this graph.

        Returns:
            A string with the graph's serialized contents.
        """
        # TODO(Chiiizzzy)
        raise NotImplementedError
