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

from towhee.dag.graph_repr import GraphRepr
from towhee.engine.graph_context import GraphContext
from towhee.dataframe import DataFrame
from towhee.engine import LOCAL_OPERATOR_CACHE


class Pipeline:
    """
    The runtime pipeline context, include graph context, all dataframes
    """

    def __init__(self, graph_repr: GraphRepr, parallelism: int = 1) -> None:
        """
        Args:
            graph_repr: (yaml `str` or `towhee.dag.GraphRepr`)
                The graph representation
            parallelism: (`int`)
                how many rows of inputs to be processed concurrently
        """
        if isinstance(graph_repr, str):
            self._graph_repr = GraphRepr.from_yaml(graph_repr)
        else:
            self._graph_repr = graph_repr

        self._parallelism = parallelism
        self.on_graph_finish_handlers = []
        self._scheduler = None

        # self.fill_cache()

        # self._cv = threading.Condition()
        # self.on_graph_finish_handlers = [self._on_graph_finish]
        # self._build()

    def register(self, scheduler):
        self._scheduler = scheduler

    def __call__(self, inputs: DataFrame) -> DataFrame:
        """
        Process one input data
        """
        if self._scheduler is None:
            raise AttributeError(
                'The pipeline is not registered to a scheduler')

        assert inputs.size == 1
        g = GraphContext(0, self._graph_repr)
        self._scheduler.register(g)
        _, data = inputs.get(0, 1)
        g(data[0])
        return g.result()

    def fill_cache(self):
        """
        Check to see if the operator directory exists locally or if needs to
        downloaded from hub
        """
        for key, value in self._graph_repr.operators.items():
            if key not in ('_start_op', '_end_op'):
                operator_path = LOCAL_OPERATOR_CACHE / (value.function)
                if operator_path.is_dir() is False:
                    print('missing', operator_path)
                else:
                    print('has', operator_path)

    # @property
    # def graph_contexts(self):
    #     return self._graph_ctxs

    # @property
    # def is_busy(self) -> bool:
    #     return False not in [graph_ctx.is_busy for graph_ctx in self._graph_ctxs]

    # def __call__(self, inputs: DataFrame) -> DataFrame:

    #     def _feed_one_graph_ctx(row):
    #         for graph_ctx in self._graph_ctxs:
    #             if not graph_ctx.is_busy:
    #                 # fill the input row to the available graph context
    #                 graph_ctx(row)
    #                 return

    #     self.outputs = DataFrame()

    #     for row in inputs.map_iter():
    #         if not self.is_busy:
    #             _feed_one_graph_ctx(row)
    #         else:
    #             with self._cv:
    #                 if not self.is_busy:
    #                     _feed_one_graph_ctx(row)
    #                 else:
    #                     self._cv.wait()

    #     # todo: GuoRentong, need to track each graph call for multi-row inputs
    #     with self._cv:
    #         if self.is_busy:
    #             self._cv.wait()
    #         return self.outputs

    # def _on_graph_finish(self, graph_ctx: GraphContext):
    #     # Organize the GraphContext's output into Pipeline's outputs.
    #     self.outputs.merge(graph_ctx.outputs)
    #     # Notify the run loop that a GraphContext is in idle state.
    #     with self._cv:
    #         self._cv.notify()

    # def _build(self):
    #     """
    #     Create GraphContexts and set up input iterators.
    #     """
    #     build graph contexts
    #     self._graph_ctxs = [GraphContext(i, self._graph_repr)
    #                         for i in range(self._parallelism)]

    #     # add on_task_finish_handlers to graph contexts
    #     for g in self._graph_ctxs:
    #         g.on_finish_handlers += self.on_graph_finish_handlers
