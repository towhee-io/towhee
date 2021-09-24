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

    def __init__(self, name: str, op_reprs: Dict[str, OperatorRepr],
                 dataframes: Dict[str, DataframeRepr]):
        self._name = name
        self._operators = op_reprs
        self._dataframes = dataframes

    @property
    def operators(self) -> Dict[str, OperatorRepr]:
        return self._operators

    @property
    def dataframes(self) -> Dict[str, DataframeRepr]:
        return self._dataframes

    @staticmethod
    def from_dict(info: Dict) -> 'GraphRepr':
        if not BaseRepr.is_valid(info, {'name', 'operators', 'dataframes'}):
            raise ValueError(
                'file or src is not a valid YAML file to describe a DAG in Towhee.'
            )
        dataframes = [DataframeRepr.from_dict(
            df_info) for df_info in info['dataframes']]
        operators = [OperatorRepr.from_dict(op_info)
                     for op_info in info['operators']]
        return GraphRepr(info['name'], operators, dataframes)

    @staticmethod
    def from_yaml(src: str) -> 'GraphRepr':
        """Import a YAML file describing this graph.

        Args:
            src(`str`):
                YAML file (could be pre-loaded as string) to import.

        example:

            name: 'test_graph'
            operators:
                -
                    name: 'test_op_1'
                    function: 'test_function'
                    inputs:
                        -
                            df: 'test_df_1'
                            col: 0
                    outputs:
                        -
                            df: 'test_df_2'
                            col: 0
                    iter_info:
                        type: map
            dataframes:
                -
                    name: 'test_df_1'
                    columns:
                        -
                            vtype: 'int'
                    name: 'test_df_2'
                    columns:
                        -
                           vtype: 'int'
        """
        info = BaseRepr.load_src(src)
        return GraphRepr.from_dict(info)

    def to_yaml(self) -> str:
        """Export a YAML file describing this graph.

        Returns:
            A string with the graph's serialized contents.
        """
        # TODO(Chiiizzzy)
        raise NotImplementedError
