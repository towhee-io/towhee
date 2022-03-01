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

import threading
from typing import Tuple

from towhee.dataframe import DataFrame
from towhee.dag import GraphRepr

from towhee.engine.operator_context import OperatorContext, OpStatus
from towhee.utils.log import engine_log
from towhee.errors import OpFailedError


class GraphContext:
    """
    `GraphContext` is a data processing network with one or multiple `Operator`.
    Each row of a `Pipeline`'s inputs will be processed individually by a
    `GraphContext`.

    Args:
        ctx_idx: (`int`)
            The index of this `GraphContext`.
        graph_repr: (`towhee.dag.GraphRepr`)
            The DAG representation this `GraphContext` will implement.
    """

    def __init__(self, ctx_idx: int, graph_repr: GraphRepr):
        self._idx = ctx_idx
        self._repr = graph_repr

        self._lock = threading.Lock()
        self._build_components()

    def __call__(self, inputs: Tuple):
        self.inputs.put(inputs)
        self.inputs.seal()

    @property
    def inputs(self) -> DataFrame:
        """Returns the graph's input `DataFrame`.

        Returns:
            (`towhee.dataframe.DataFrame`)
                The input `DataFrame`.
        """
        return self.dataframes['_start_df']

    @property
    def outputs(self) -> DataFrame:
        """Returns the graph's output `DataFrame`.

        Returns:
            (`towhee.dataframe.DataFrame`)
                The output `DataFrame`.
        """
        return self.dataframes['_end_df']

    def result(self) -> any:
        if self.outputs.size != 0:
            return self.outputs
        else:
            # graph run failed, raise an exception
            for op in self._op_ctxs.values():
                if op.status == OpStatus.FAILED:
                    raise OpFailedError(op.err_msg)
            engine_log.warning('The pipeline runs successfully, but no data return')
            return None

    @property
    def op_ctxs(self):
        return self._op_ctxs

    @property
    def dataframes(self):
        return self._dataframes

    def slow_down(self, df_name: str, time_sec: int):
        '''
        Slow down the op whose df name it.
        '''
        self._df_op[df_name].slow_down(time_sec)

    def speed_up(self, df_name: str):
        '''
        spped up the op whose df name it.
        '''
        self._df_op[df_name].speed_up()

    def stop(self):
        for op in self._op_ctxs:
            op.stop()

    def gc(self):
        for _, df in self._dataframes.items():
            df.gc_data()

    def join(self):
        for op in self._op_ctxs.values():
            op.join()

    def _build_components(self):
        """
        Builds `DataFrame`s and `OperatorContext`s required to run this graph.
        """

        # Build dataframes.
        dfs = {}
        for df_name, df_repr in self._repr.dataframes.items():
            cols = [(col.name, col.vtype) for col in df_repr.columns]
            dfs[df_name] = DataFrame(df_name, cols)
        self._dataframes = dfs

        # Build operator contexts.
        self._op_ctxs = {}
        self._df_op = {}
        for _, op_repr in self._repr.operators.items():
            op_ctx = OperatorContext(op_repr, self.dataframes)
            self._df_op[op_repr.outputs[0]['df']] = op_ctx
            self._op_ctxs[op_ctx.name] = op_ctx

    def __del__(self):
        engine_log.info('Graph % s end', self._idx)
