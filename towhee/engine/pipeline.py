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
from towhee.dataframe import DataFrame
from towhee.engine.graph_context import GraphContext
from towhee.engine.task_scheduler import TaskScheduler


class Pipeline:
    """
    The runtime pipeline context, include graph context, all dataframes.

    Args:
        graph_repr: (`str` or `towhee.dag.GraphRepr`)
            The graph representation either as a YAML-formatted string, or directly
            as an instance of `GraphRepr`.
        parallelism: (`int`)
            The parallelism parameter dictates how many copies of the graph context
            we create. This is likely a low number (1-4) for local engines, but may
            be much higher for cloud instances.
    """

    def __init__(self, graph_repr: GraphRepr, parallelism: int = 1) -> None:

        self._parallelism = parallelism
        self.on_graph_finish_handlers = []
        self._scheduler = None

        # TODO(fzliu): instantiate `DataFrame` columns.
        self._outputs = None

        # Instantiate the graph representation.
        if isinstance(graph_repr, str):
            self._graph_repr = GraphRepr.from_yaml(graph_repr)
        else:
            self._graph_repr = graph_repr

        # self._cv = threading.Condition()
        # self.on_graph_finish_handlers = [self._on_graph_finish]
        # self._build()

    @property
    def parallelism(self) -> int:
        return self._parallelism

    @parallelism.setter
    def parallelism(self, val):
        if isinstance(val, int) and val > 0:
            self._parallelism = val
        else:
            raise ValueError('Parallelism value must be a positive integer')

    @property
    def outputs(self) -> DataFrame:
        return self._outputs

    def register(self, scheduler: TaskScheduler):
        self._scheduler = scheduler

    def __call__(self, inputs: DataFrame) -> DataFrame:
        """
        Process an input `DataFrame`. This function instantiates an output `DataFrame`;
        upon completion, individual `GraphContext` outputs are merged into this
        dataframe. Inputs are weaved through the input `DataFrame` for each
        `GraphContext` as follows (`parallelism = 3`):
            data[0] -> ctx[0]
            data[1] -> ctx[1]
            data[2] -> ctx[2]
            data[3] -> ctx[0]
            data[4] -> ctx[1]

        Args:
            inputs (`towhee.dataframe.DataFrame`):
                Input `DataFrame` (with potentially multiple rows) to process.

        Returns:
            (`towhee.dataframe.DataFrame`)
                Output `DataFrame` with ordering matching the input `DataFrame`.
        """
        if not self._scheduler:
            raise AttributeError('Pipeline not registered to a Scheduler.')

        graph_ctxs = []
        for n in range(self._parallelism):
            graph_ctx = GraphContext(n, self._graph_repr)
            self._scheduler.register(graph_ctx)
            graph_ctxs.append(graph_ctx)

        self._outputs = DataFrame()

        # Weave inputs into each GraphContext.
        # TODO(fzliu): there are better ways to maintain ordering.
        for n, row in enumerate(inputs.map_iter()):
            idx = n % len(graph_ctxs)
            graph_ctxs[idx](row[0])

        for graph_ctx in graph_ctxs:
            graph_ctx.inputs.seal()
            graph_ctx.outputs.wait_sealed()

        # TODO(fzliu): Create an operator to merge multiple `DataFrame` instances.
        idx = 0
        merge = True
        while merge:
            for graph_ctx in graph_ctxs:
                elem = graph_ctx.outputs.get(idx, 1)
                if not elem:
                    merge = False
                    break
                self._outputs.put(elem[0])
            idx += 1

        return self._outputs

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
