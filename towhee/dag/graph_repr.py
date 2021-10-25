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
from typing import Dict, List, Set, Tuple, Any

from towhee.dag.base_repr import BaseRepr
from towhee.dag.dataframe_repr import DataFrameRepr
from towhee.dag.operator_repr import OperatorRepr


class GraphRepr(BaseRepr):
    """
    A `GraphRepr` presents a complete DAG.

    A graph contains individual subcomponents, including Operators, Dataframes, and
    Variables. Graph representations are used during execution to load functions and
    pass data to the correct operators.

    Args:
        name (`str`):
            The representation name.
        file_or_url (`str`):
            The file or remote url that stores the information of this representation.
    """
    def __init__(self, name: str, op_reprs: Dict[str, OperatorRepr], df_reprs: Dict[str, DataFrameRepr]):
        super().__init__(name)
        self._operators = op_reprs
        self._dataframes = df_reprs

    @property
    def operators(self) -> Dict[str, OperatorRepr]:
        return self._operators

    @property
    def dataframes(self) -> Dict[str, DataFrameRepr]:
        return self._dataframes

    @staticmethod
    def dfs(cur: str, adj: Dict[str, List[str]], flag: Dict[str, int], cur_list: List[str]) -> Tuple[bool, List[str]]:
        """
        Depth-First Search the graph.

        Args:
            cur (`str`):
                The name of current dataframe.
            adj (`Dict[str, List[str]]`):
                A dict store the adjacent dataframe of each dataframe.
            flag (`Dict[str, int]`):
                A dict store the status of the columns.
                - 0 means the dataframe has not been visted yet.
                - 1 means the dataframe has been visted in this search, which means this
                columns is part of a loop.
                - 2 means the dataframe has been searched and confirmed not to be a part
                of a loop.
            cur_list (`List[str]`):
                The list of dataframe that have been visited in this search.

        Returns:
            (`Tuple[bool, List[str]]`)
                Return `False` if there is no loop, else `True` and the loop.
        """
        # If `cur` is not the input of any operator, it will not form a loop anyway.
        if cur not in flag:
            return False, []
        # If `cur` has been searched and not in a loop.
        if flag[cur] == 2:
            return False, []
        # If `cur` has been visited in this search.
        if flag[cur] == 1:
            return True, cur_list
        flag[cur] = 1
        # Recursion
        for col in adj[cur]:
            cur_list.append(cur)
            status, loop_list = GraphRepr.dfs(col, adj, flag, cur_list)
            if status:
                return status, loop_list
        flag[cur] = 2

        return False, []

    def get_loop(self) -> List[str]:
        """
        Get the loop(s) inside the graph.

        Returns:
            (`List[str]`)
                Return the loop if exists, else an empty list.
        """
        adj = {}
        flag = {}
        dataframes = []
        # Collect adjacent information and mark all the flag as 0.
        for op in self._operators.values():
            out = op.outputs[0]['df']
            for df in op.inputs:
                name = df['df']
                if name not in dataframes:
                    adj[name] = []
                    flag[name] = 0
                    dataframes.append(name)
                adj[name].append(out)
        # Recursive DFS.
        for df in dataframes:
            status, loop_list = GraphRepr.dfs(df, adj, flag, [])
            if status:
                return loop_list

        return []

    def get_isolated_df(self) -> Set[str]:
        """
        Get the isolated dataframe(s) in the DAG.

        Returns:
            (`Set[str]`)
                Return the isolated dataframe set if exists, else an empty set.
        """
        # First mark all the dataframes as isolated.
        iso_df = {df.name for df in self._dataframes.values()}
        # If the dataframe is the input/output of any operators remove it from isolated dataframes set.
        for op in self._operators.values():
            for i in op.inputs:
                iso_df.discard(i['df'])
            iso_df.discard(op.outputs[0]['df'])

        return iso_df

    def get_isolated_op(self) -> Set[str]:
        """
        Get the isolated operator(s) in the DAG.

        Returns:
            (`Set[str]`)
                Return the isolated operator set if exists, else an empty set.
        """
        # If the graph has only one operator, there is no isolated operator.
        if len(self._operators.values()) == 1:
            return set()

        in_df = set()
        out_df = set()
        iso_op = set()
        # Collect all the input/output dataframes.
        for op in self._operators.values():
            for i in op.inputs:
                in_df.add(i['df'])
            out_df.add(op.outputs[0]['df'])
        # Traverse the operators to find isolated operators.
        for op in self._operators.values():
            cur_in = {i['df'] for i in op.inputs}
            cur_out = {i['df'] for i in op.outputs}
            # If the input/output are not the output/input of any other operators the operator is isolated.
            if not cur_in.intersection(out_df - cur_out) and not cur_out.intersection(in_df - cur_in):
                iso_op.add(op.name)

        return iso_op

    @staticmethod
    def from_dict(info: Dict[Any, Any]) -> 'GraphRepr':
        """
        Generate a GraphRepr from a description dict.

        Args:
            info (`Dict[Any, Any]`):
                A dict to describe the DAG.

        Returns:
            (`towhee.dag.GraphRepr`)
                The GraphRepr obj.
        """
        # Basic schema check.
        if not BaseRepr.is_valid(info, {'name', 'operators', 'dataframes'}):
            raise ValueError('file or src is not a valid YAML file to describe a DAG in Towhee.')
        dataframes = dict((df_info['name'], DataFrameRepr.from_dict(df_info)) for df_info in info['dataframes'])
        operators = dict((op_info['name'], OperatorRepr.from_dict(op_info)) for op_info in info['operators'])
        return GraphRepr(info['name'], operators, dataframes)

    @staticmethod
    def from_yaml(src: str):
        """
        Import a YAML file describing this graph.
        Example YAML look like this:
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

        Args:
            src (`str`):
                YAML file (could be pre-loaded as string) to import.

        Returns:
            (`towhee.dag.GraphRepr`)
                The GraphRepr object.
        """
        info = BaseRepr.load_src(src)
        return GraphRepr.from_dict(info)
        # Load the information and get isolation and loop.
        # info = BaseRepr.load_src(src)
        # graph = GraphRepr.from_dict(info)
        # iso_df = graph.get_isolated_df()
        # iso_op = graph.get_isolated_op()
        # loop = graph.get_loop()
        # # If no loop or isolation found, return the GraphRepr
        # if not iso_df and not iso_op and not loop:
        #     return GraphRepr.from_dict(info)
        # # Generate error message.
        # df_msg = '' if not iso_df else f'The DAG contains isolated dataframe(s) {iso_df}.'
        # op_msg = '' if not iso_op else f'The DAG contains isolated operator(s) {iso_op}.'
        # loop_msg = '' if not loop else f'The DAG contains loop consists of {loop}'
        # msg = '.'.join([df_msg, op_msg, loop_msg]).split(None)

        # raise ValueError(msg)

    def to_yaml(self) -> str:
        """Export a YAML file describing this graph.

        Returns:
            (`str`)
                A string with the graph's serialized contents.
        """
        # TODO(Chiiizzzy)
        raise NotImplementedError
