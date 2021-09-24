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

from typing import List, Dict, Tuple


from towhee.dag.graph_repr import GraphRepr
from towhee.dag.operator_repr import OperatorRepr
from towhee.dataframe import DataFrame
from towhee.engine.graph_context import GraphContext
from towhee.engine.operator_context import OperatorContext, OpInfo


class _ReprToCtx:
    """
    Representation to graph_ctx, op_ctx, dataframes
    """

    def __init__(self, graph_yaml: str):
        self._graph_repr = GraphRepr.from_yaml(graph_yaml)
        self._dataframes = None
        self._graph_ctx = None
        self._op_ctxs = None

    @property
    def dataframes(self) -> Dict[str, DataFrame]:
        if self._dataframes is None:
            frames = {}
            for df_repr in self._graph_repr.dataframes:
                cols = {}
                for i in range(len(df_repr.columns)):
                    col = df_repr.columns[i]
                    cols[col.name] = {}
                    cols[col.name]['index'] = i
                    cols[col.name]['type'] = col.vtype
                frames[df_repr.name] = DataFrame(df_repr.name, cols)
            self._dataframes = frames
        return self._dataframes

    @property
    def op_ctxs(self) -> List[OperatorContext]:
        if self._op_ctxs is None:
            op_ctxs = []
            for op_repr in self._graph_repr.operators:
                op_ctxs.append(self._build_op_ctx(op_repr))
            self._op_ctxs = op_ctxs
        return self._op_ctxs

    @property
    def graph_ctx(self) -> GraphContext:
        if self._graph_ctx is None:
            self._graph_ctx = GraphContext(self.op_ctxs)
        return self._graph_ctx

    def _build_op_ctx(self, op_repr: OperatorRepr) -> OperatorContext:
        df_in = [self.dataframes[df_info['df']] for df_info in op_repr.inputs]
        df_out = [self.dataframes[df_info['df']]
                  for df_info in op_repr.outputs]

        return OperatorContext(
            OpInfo(op_repr.name, op_repr.function, op_repr.init_args,
                   op_repr.iter_info['type'],
                   dict((item['name'], item['col']) for item in op_repr.inputs)),
            df_in,
            df_out
        )


def create_ctxs(graph_yaml) -> Tuple[GraphContext, List[DataFrame]]:
    obj = _ReprToCtx(graph_yaml)
    return obj.graph_ctx, obj.dataframes
