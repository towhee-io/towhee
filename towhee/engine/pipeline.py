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
from towhee.engine import LOCAL_OPERATOR_CACHE
from towhee.engine.graph_context import GraphContext
from towhee.engine.task_scheduler import TaskScheduler


class Pipeline:
    """
    The runtime pipeline context, include graph context, all dataframes
    """

    def __init__(self, graph_repr: GraphRepr, parallelism: int = 1) -> None:
        """
        Args:
            graph_repr: (`str` or `towhee.dag.GraphRepr`)
                The graph representation either as a YAML-formatted string, or directly
                as an instance of `GraphRepr`.
            parallelism: (`int`)
                how many rows of inputs to be processed concurrently
        """

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

        # self.fill_cache()

        # self._cv = threading.Condition()
        # self.on_graph_finish_handlers = [self._on_graph_finish]
        # self._build()

    @property
    def outputs(self) -> DataFrame:
        return self._outputs

    def register(self, scheduler: TaskScheduler):
        self._scheduler = scheduler

    def __call__(self, inputs: DataFrame) -> DataFrame:
        """Process an input `DataFrame`.

        Args:
            inputs: (`towhee.DataFrame`)
        """
        # Pipelines must register themselves with schedulers.
        if not self._scheduler:
            raise AttributeError('Pipeline not registered to a Scheduler.')

        self._outputs = DataFrame()

        # The parallelism parameter dictates how many copies of the graph context we
        # create. This is likely a low number (1-4) for local engines, but may be much
        # higher for cloud instances.
        graph_ctxs = []
        for n in range(self._parallelism):
            graph_ctx = GraphContext(n, self._graph_repr)
            self._scheduler.register(graph_ctx)
            graph_ctxs.append(graph_ctx)

        # Weave inputs into each GraphContext.
        # TODO(fzliu): can we spin off a thread to do this?
        for n, row in enumerate(inputs.map_iter()):
            idx = n % len(graph_ctxs)
            graph_ctxs[idx](row[0])

        # Instantiate an output `DataFrame`. Upon completion, individual `GraphContext`
        # outputs are merged into this dataframe.
        # TODO(fzliu): Create an operator to merge multiple `DataFrame` instances.
        for graph_ctx in graph_ctxs:
            graph_ctx.inputs.seal()
            graph_ctx.outputs.wait_sealed()
            #print(graph_ctx.outputs.size)
            self._outputs.merge(graph_ctx.outputs)

        return self._outputs

    def fill_cache(self):
        """Check to see if the operator directory exists locally or if needs to
        be downloaded from the hub.
        """
        for key, value in self._graph_repr.operators.items():
            if key not in ('_start_op', '_end_op'):
                operator_path = LOCAL_OPERATOR_CACHE / (value.function)

                # TODO(filip-halt): replace prints with code.
                if operator_path.is_dir() is False:
                    print('missing', operator_path)
                else:
                    print('has', operator_path)

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
