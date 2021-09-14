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

from towhee.base_repr import BaseRepr
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
    def operators(self, name: str, value: OperatorRepr):
        self._operators[name] = value

    @dataframes.setter
    def dataframes(self, name: str, value: DataframeRepr):
        self._dataframes[name] = value

    @staticmethod
    def load_file(file_or_src: str):
        # we support file from local file/HTTP/HDFS
        # if `file_or_src` is a loacl file
        if os.path.isfile(file_or_src):
            with open(file_or_src, 'r', encoding='utf-8') as f:
                src = yaml.safe_dump(yaml.safe_load(f), default_flow_style=False)
            return src
        # if `file_or_src` from HTTP
        elif file_or_src.lower().startswith('http'):
            src = requests.get(file_or_src).content.decode('utf-8')
            return src
        # if `file_or_src` is YAMl (pre-loaded as str)
        return file_or_src

    @staticmethod
    def from_yaml(file_or_src: str):
        """Import a YAML file describing this graph.

        Args:
            src:
                YAML file (pre-loaded as string) to import.
        """
        graph = GraphRepr()
        src = graph.load_file(file_or_src)

        # load YAML file as a dictionray
        graph_dict = yaml.safe_load(src)

        # some simple schema checks for YAML file
        essentials = {'graph', 'operators', 'dataframes'}
        if not isinstance(graph_dict, dict):
            raise ValueError('src is not a valid YAML file')
        if not essentials.issubset(set(graph_dict.keys())):
            raise ValueError('src cannot descirbe a DAG in Towhee')

        # load name
        graph.name = graph_dict['graph']['name']

        # load dataframes
        for df in graph_dict['dataframes']:
            graph.dataframes[df['name']] = DataframeRepr.from_yaml(yaml.safe_dump(df, default_flow_style=False))

        # load operators
        for op in graph_dict['operators']:
            graph.operators[op['name']] = OperatorRepr.from_yaml(yaml.safe_dump(op, default_flow_style=False), graph.dataframes)

        return graph

    def to_yaml(self) -> str:
        """Export a YAML file describing this graph.

        Returns:
            A string with the graph's serialized contents.
        """
        # TODO(Chiiizzzy)
        raise NotImplementedError
