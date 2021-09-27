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
from typing import Dict, List, Tuple

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
    def __init__(self, name: str, op_reprs: Dict[str, OperatorRepr], df_reprs: Dict[str, DataframeRepr]):
        super().__init__(name)
        self._operators = op_reprs
        self._dataframes = df_reprs

    @property
    def operators(self) -> Dict[str, OperatorRepr]:
        return self._operators

    @property
    def dataframes(self) -> Dict[str, DataframeRepr]:
        return self._dataframes

    @staticmethod
    def dfs(cur: str, adj: Dict[str, List[str]], flag: Dict[str, int], cur_list: List[str]) -> Tuple[bool, str]:
        """Depth-First Search the graph
        Args:
            cur(`str`):
                The name of current dataframe.
            adj(`dict`):
                A dict store the adjacent dataframe of each dataframe.
            flag(`dict`):
                A dict store the status of the columns.
                - 0 means the dataframe has not been visted yet.
                - 1 means the dataframe has been visted in this search, which means this
                columns is part of a loop.
                - 2 means the dataframe has been searched and confirmed not to be a part
                of a loop.
            cur_list('list):
                The list of dataframe that have been visited in this search.
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
            return True, f'The dataframes {cur_list} forms a loop.'
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
        dataframes = []

        for op in self._operators:
            out = op.outputs[0]['df']
            for df in op.inputs:
                name = df['df']
                if name not in dataframes:
                    adj[name] = []
                    flag[name] = 0
                    dataframes.append(name)
                adj[name].append(out)

        # Recursive DFS
        for df in dataframes:
            status, msg = GraphRepr.dfs(df, adj, flag, [])
            if status:
                return status, msg

        return False, ''

    def has_isolated_df(self):
        iso_df = {df.name for df in self._dataframes}

        for op in self._operators:
            for i in op.inputs:
                iso_df.discard(i['df'])
            iso_df.discard(op.outputs[0]['df'])

        status = bool(iso_df)
        msg = '' if not iso_df else f'The DAG contains isolated dataframe(s) {iso_df}.'

        return status, msg

    def has_isolated_op(self):
        if len(self._operators) == 1:
            return False, ''

        in_df = set()
        out_df = set()
        iso_op = set()

        for op in self._operators:
            for i in op.inputs:
                in_df.add(i['df'])
            out_df.add(op.outputs[0]['df'])

        for op in self._operators:
            cur_in = {i['df'] for i in op.inputs}
            cur_out = {i['df'] for i in op.outputs}

            if not cur_in.intersection(out_df - cur_out) and not cur_out.intersection(in_df - cur_in):
                iso_op.add(op.name)

        status = bool(iso_op)
        msg = '' if not iso_op else f'The DAG contains isolated operator(s) {iso_op}.'

        return status, msg

    @staticmethod
    def from_dict(info: Dict) -> 'GraphRepr':
        if not BaseRepr.is_valid(info, {'name', 'operators', 'dataframes'}):
            raise ValueError('file or src is not a valid YAML file to describe a DAG in Towhee.')
        dataframes = [DataframeRepr.from_dict(df_info) for df_info in info['dataframes']]
        operators = [OperatorRepr.from_dict(op_info) for op_info in info['operators']]
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
