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


# from towhee.engine.pipeline import Pipeline

from typing import Tuple
import threading

from towhee.dataframe import DataFrame
from towhee.dag.graph_repr import GraphRepr


from towhee.engine.operator_context import OperatorContext


class GraphContext:
    """
    `GraphContext` is a data processing network with one or multiple `Operator`.
    Each row of a `Pipeline`'s inputs will be processed individually by a
    `GraphContext`.

    Args:
        ctx_idx: (int)
            The index of this `GraphContext`
        graph_repr: (`towhee.dag.GraphRepr`)
            The DAG representation
    """

    def __init__(self, ctx_idx: int, graph_repr: GraphRepr):
        self._idx = ctx_idx
        self._repr = graph_repr
        self._mutex = threading.Lock()
        self._finished = threading.Condition(self._mutex)

        self.on_start_handlers = []
        self.on_finish_handlers = []

        self.on_task_ready_handlers = []
        self.on_task_start_handlers = []
        # self.on_task_finish_handlers = [self._on_task_finish]
        self.on_task_finish_handlers = []

        self._build()
        self._cv = threading.Condition()

    def __call__(self, inputs: Tuple):
        graph_input = self.operator_contexts['_start_op'].outputs[0]
        graph_input.put(inputs)
        graph_input.seal()

    def result(self):
        output_df = self.operator_contexts['_end_op'].inputs[0]
        output_df.wait_sealed()
        return output_df

    # def __call__(self, inputs: Tuple):
    #     # todo: GuoRentong, issue #114
    #     graph_input = self.operator_contexts['_start_op'].inputs[0]
    #     graph_input.clear()
    #     graph_input.put(inputs)
    #     graph_input.seal()
    #     self._is_busy = True

    @property
    def operator_contexts(self):
        return self._op_contexts

    @property
    def dataframes(self):
        return self._dataframes

    # @property
    # def outputs(self) -> DataFrame:
    #     return self.operator_contexts['_end_op'].outputs[0]

    # @property
    # def is_busy(self):
    #     return self._is_busy

    # def _on_task_finish(self, task: Task):
    #     """
    #     Callback after the execution of a `Task`.
    #     """
    #     # todo: GuoRentong, currently, each `GraphContext` can only process one row
    #     # the following finish condition is ugly :(
    #     if self.operator_contexts[task.op_name].outputs[0] is self.outputs:
    #         self._is_busy = False

    #     if not self._is_busy:
    #         # call on_finish_handlers
    #         for handler in self.on_finish_handlers:
    #             handler(self)

    def _build(self):
        # build dataframes
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

        # build operator contexts
        self._op_contexts = {}
        for _, op_repr in self._repr.operators.items():
            is_schedulable = self._is_schedulable_op(op_repr)
            op_ctx = OperatorContext(op_repr, self.dataframes, is_schedulable)

            op_ctx.on_task_start_handlers += self.on_task_start_handlers
            op_ctx.on_task_ready_handlers += self.on_task_ready_handlers
            op_ctx.on_task_finish_handlers += self.on_task_finish_handlers

            self._op_contexts[op_ctx.name] = op_ctx

    @staticmethod
    def _is_schedulable_op(op_repr):
        op_name = op_repr.name
        if op_name in ('_start_op', '_end_op'):
            return False
        else:
            return True
