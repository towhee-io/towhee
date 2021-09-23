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
from typing import Dict, List, Tuple

from towhee.dag.base_repr import BaseRepr
from towhee.dag.dataframe_repr import DataframeRepr
from towhee.dag.operator_repr import OperatorRepr


class GraphRepr(BaseRepr):
    """This class presents a complete representation of a graph.

    A graph contains individual subcomponents, including Operators, Dataframes, and
    Variables. Graph representations are used during execution to load functions and
    pass data to the correct operators.
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
    def dfs(cur: str, adj: Dict[str, List[str]], flag: Dict[str, int], cur_list: List[str]) -> Tuple[bool, str]:
        """Depth-First Search the graph

        Args:
            cur(`str`):
                The name of current column.
            adj(`dict`):
                A dict store the adjacent columns of each column.
            flag(`dict`):
                A dict store the status of the columns.
                - 0 means the column has not been visted yet.
                - 1 means the column has been visted in this search, which means this
                columns is part of a loop.
                - 2 means the column has been searched and confirmed not to be a part
                of a loop.
            cur_list('list):
                The list of columns that have been visited in this search.

        Returns:
            Return `False` if there is no loop, else `True` and the loop message.
        """
        # if `cur` is not the input of any operator, it will not form a loop anyway
        if cur not in flag:
            return False, ''
        # If `cur` has been searched and not in a loop
        if flag[cur] == 2:
            return False, ''
        # If `cur` has been visited in this search
        if flag[cur] == 1:
            return True, f'The columns {cur_list} forms a loop'
        flag[cur] = 1

        for col in adj[cur]:
            cur_list.append(cur)
            status, msg = GraphRepr.dfs(col, adj, flag, cur_list)
            if status:
                return status, msg
        flag[cur] = 2

        return False, ''

    def has_loop(self) -> Tuple[bool, str]:
        """Check if there are loop(s) inside the graph.

        Returns:
            Return `False` if there is no loop, else `True` and the loop message.
        """
        adj = {}
        flag = {}
        cols = []

        for op in self._operators.values():
            out = list(op.outputs)[0]
            for col in op.inputs:
                if col not in cols:
                    adj[col] = []
                    flag[col] = 0
                    cols.append(col)
                adj[col].append(out)

        for col in cols:
            status, msg = self.dfs(col, adj, flag, [])
            if status:
                return status, msg

        return False, ''

    def has_isolated_op(self) -> Tuple[bool, str]:
        """Check if there are isolated operators in the DAG.

        Returns:
            Return `False` if there is no isolated operator.
            Else `True` and the error message.
        """
        iso_op = set()
        in_col = set()
        out_col = set()

        # gather non-isolated columns
        for op in self._operators.values():
            for col in op.inputs:
                in_col.add(col)
            for col in op.outputs:
                out_col.add(col)

        for op in self._operators.values():
            cur_in = set(op.inputs)
            cur_out = set(op.outputs)
            if not cur_in.intersection(out_col) and not cur_out.intersection(in_col):
                iso_op.add(op.name)

        status = bool(iso_op)
        msg = '' if not iso_op else f'The DAG contains isolated operator(s): {iso_op}'

        return status, msg

    def has_isolated_df(self) -> Tuple[bool, str]:
        """Check if there are isolated dataframes in the DAG.

        Returns:
            Return `False` if there is no isolated dataframe.
            Else `True` and the error message.
        """
        if len(self._operators) == 1:
            return False, ''
        # reset
        in_df = set()
        out_df = set()
        iso_df = set()

        # gather non-isolated dataframes
        for op in self._operators.values():
            for df in op.inputs.values():
                in_df.add(df['df'].name)
            for df in op.outputs.values():
                out_df.add(df['df'].name)

        # If a dataframe is not the input or output of any operators
        for df in self._dataframes:
            if df not in in_df.union(out_df):
                iso_df.add(df)

        status = bool(iso_df)
        msg = '' if not iso_df else f'The DAG contains isolated dataframe(s): {iso_df}'

        return status, msg

    def has_isolation(self) -> Tuple[bool, str]:
        """Check if there are isolated components(operators and dataframes) in the DAG.

        Returns:
            Return `False` if there is no isolated components.
            Else `True` and the error mesasge.
        """
        df_status, df_msg = self.has_isolated_df()
        op_status, op_msg = self.has_isolated_op()

        status = df_status or op_status
        msg = '.'.join(filter(None, [df_msg, op_msg]))

        return status, msg

    @staticmethod
    def is_valid(info: List[dict]) -> bool:
        """Check if `info` contains the valid inforamtion to describe a DAG in Towhee.

        Args:
            info(`list`):
                The List loaded from the source file.

        Returns:
            Return `True` if the src decribes a valid DAG in Towhee, else `False`.
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
        """Import a YAML file describing the graph.

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

        # The DAG is supposed not to include loops, isolated operators and dataframes.
        iso_status, iso_msg = graph_repr.has_isolation()
        loop_status, loop_msg = graph_repr.has_loop()
        status = iso_status or loop_status
        msg = '.'.join(filter(None, [iso_msg, loop_msg]))
        if status:
            # ValueError or what?
            raise ValueError(msg)

        return graph_repr

    def to_yaml(self) -> str:
        """Export a YAML file describing this graph.

        Returns:
            A string with the graph's serialized contents.
        """
        # TODO(Chiiizzzy)
        raise NotImplementedError
