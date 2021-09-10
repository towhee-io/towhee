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
    """
    def __init__(self, name: str, file_or_url: str = None):
        super().__init__(name)
        self._operators = {}
        self._dataframes = {}
        # if `file` is a loacl file
        if file_or_url and os.path.isfile(file_or_url):
            with open(file_or_url, 'r', encoding='utf-8') as f:
                self._src = yaml.safe_load(f)
            self.from_yaml(self._src)
        # if `file` is a remote url
        elif file_or_url:
            self._src = requests.get(file_or_url).content.decode('utf-8')
            self.from_yaml(self._src)

    @property
    def operators(self) -> Dict[str, OperatorRepr]:
        return self._operators

    @property
    def dataframes(self) -> Dict[str, DataframeRepr]:
        return self._dataframes

    def from_yaml(self, src: str):
        """Import a YAML file describing this graph.

        Args:
            src:
                YAML file (pre-loaded as string) to import.
        """
        # load YAML file as a dictionray
        graph_dict = yaml.safe_load(src)

        # some simple schema checks for YAML file
        essentials = {'graph', 'operators', 'dataframes'}
        if not isinstance(graph_dict, dict):
            raise ValueError('src is not a valid YAML file')
        if not essentials.issubset(set(graph_dict.keys())):
            raise ValueError('src cannot descirbe a DAG in Towhee')

        # load name from YAML
        self._name = graph_dict['graph']['name']

        # load dataframes from YAML
        for df in graph_dict['dataframes']:
            self._dataframes[df['name']] = DataframeRepr(df['name'], str(df))

        # load operators from YAML
        for op in graph_dict['operators']:
            self._operators[op['name']] = OperatorRepr(op['name'], self._dataframes, str(op))

    def to_yaml(self) -> str:
        """Export a YAML file describing this graph.

        Returns:
            A string with the graph's serialized contents.
        """
        # TODO(Chiiizzzy)
        raise NotImplementedError
