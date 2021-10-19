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

from towhee.dataframe import DataFrame, Variable
from towhee.dag import GraphRepr
from towhee.engine.task import Task
from towhee.utils import HandlerMixin

from towhee.engine.operator_context import OperatorContext


class GraphContext(HandlerMixin):
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

        self._is_busy = False
        self._lock = threading.Lock()
        self._done_cv = threading.Condition(self._lock)

        self.add_handler_methods(
            'graph_start',
            'graph_finish',
            'task_ready',
            'task_start',
            'task_finish'
        )

        self._build_components()

        self.add_graph_start_handler(self._on_graph_start)
        self.add_graph_finish_handler(self._on_graph_finish)

    def __call__(self, inputs: Tuple[Variable]):
        self.inputs.put(inputs)
        #graph_input = self.op_ctxs['_start_op'].outputs[0]
        #graph_input.put(inputs)
        #graph_input.seal()

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

    # def result(self):
    #     output_df = self.op_ctxs['_end_op'].inputs[0]
    #     output_df.wait_sealed()
    #     return output_df

    # def __call__(self, inputs: Tuple):
    #     # todo: GuoRentong, issue #114
    #     graph_input = self.op_ctxs['_start_op'].inputs[0]
    #     graph_input.clear()
    #     graph_input.put(inputs)
    #     graph_input.seal()
    #     self._is_busy = True

    @property
    def op_ctxs(self):
        return self._op_ctxs

    @property
    def dataframes(self):
        return self._dataframes

    @property
    def is_busy(self):
        return self._is_busy

    def wait_done(self):
        with self._done_cv:
            if self._is_busy:
                self._done_cv.wait()

    def reset(self):
        """Resets `OperatorContext` instances which maintain state within the graph.
        Stateless operators are left unchanged.
        """
        raise NotImplementedError

    def _on_graph_start(self, task: Task):
        if task.hub_op_id == '_start_op':
            with self._done_cv:
                self._is_busy = True

    def _on_graph_finish(self, task: Task):
        if task.hub_op_id == '_end_op':
            with self._done_cv:
                self._is_busy = False
                self._done_cv.notify_all()

    # def _on_task_finish(self, task: Task):
    #     """
    #     Callback after the execution of a `Task`.
    #     """
    #     # todo: GuoRentong, currently, each `GraphContext` can only process one row
    #     # the following finish condition is ugly :(
    #     if self.op_ctxs[task.op_name].outputs[0] is self.outputs:
    #         self._is_busy = False

    #     if not self._is_busy:
    #         # call on_finish_handlers
    #         for handler in self.on_finish_handlers:
    #             handler(self)

    def _build_components(self):
        """Builds `DataFrame`s and `OperatorContext`s required to run this graph.
        """

        # Build dataframes.
        dfs = {}
        for df_name, df_repr in self._repr.dataframes.items():
            cols = {}
            for i in range(len(df_repr.columns)):
                col = df_repr.columns[i]
                cols[col.name] = {}
                cols[col.name]['index'] = i
                cols[col.name]['type'] = col.vtype
            dfs[df_name] = DataFrame(df_name, cols)
        self._dataframes = dfs

        # Build operator contexts.
        self._op_ctxs = {}
        for _, op_repr in self._repr.operators.items():
            op_ctx = OperatorContext(op_repr, self.dataframes)

            if op_ctx.name == '_start_op':
                op_ctx.add_task_start_handler(self.graph_start_handlers)
            elif op_ctx.name == '_end_op':
                op_ctx.add_task_finish_handler(self.graph_finish_handlers)
            else:
                op_ctx.add_task_start_handler(self.task_start_handlers)
                op_ctx.add_task_ready_handler(self.task_ready_handlers)
                op_ctx.add_task_finish_handler(self.task_finish_handlers)

            self._op_ctxs[op_ctx.name] = op_ctx
