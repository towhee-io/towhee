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
import logging
from typing import Dict, List

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
    def is_valid(info: List[dict]) -> bool:
        """Check if the src is a valid YAML file to describe a DAG in Towhee.

        Args:
            info(`list`):
                The List loaded from the source file.
        """
        essentials = {'graph', 'operators', 'dataframes'}
        if not isinstance(info, list):
            logging.error('src is not a valid YAML file.')
            return False
        for i in info:
            if not isinstance(i, dict):
                logging.error('src is not a valid YAML file.')
                return False
            if not essentials.issubset(set(i.keys())):
                logging.error('src cannot descirbe a DAG in Towhee.')
                return False
        return True

    @staticmethod
    def from_yaml(file_or_src: str):
        """Import a YAML file describing this graph.

        Args:
            file_or_src(`str`):
                YAML file (could be pre-loaded as string) to import.

        Returns:
            The DAG we described in `file_or_src`.
        """
        graph_repr = GraphRepr()
        graphs = graph_repr.load_src(file_or_src)
        graph = graphs[0]
        if not graph_repr.is_valid(graphs):
            raise ValueError('file or src is not a valid YAML file to describe a DAG in Towhee.')

        # load name
        graph_repr.name = graph['graph']['name']

        # load dataframes
        graph_repr.dataframes = DataframeRepr.from_yaml(yaml.safe_dump(graph['dataframes'], default_flow_style=False))

        # load operators
        graph_repr.operators = OperatorRepr.from_yaml(yaml.safe_dump(graph['operators'], default_flow_style=False), graph_repr.dataframes)

        return graph_repr

    def to_yaml(self) -> str:
        """Export a YAML file describing this graph.

        Returns:
            A string with the graph's serialized contents.
        """
        # TODO(Chiiizzzy)
        raise NotImplementedError
