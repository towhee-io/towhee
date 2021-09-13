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
from typing import Callable

from towhee.dag.graph_repr import GraphRepr
from towhee.dataframe.dataframe import DFIterator
from towhee.engine.graph_context import GraphContext


class Pipeline(threading.Thread):
    """
    The runtime pipeline context
    """

    def __init__(self, graph_repr: GraphRepr, parallelism: int = 1) -> None:
        """
        Args:
            graph_repr: the graph representation
            parallelism: how many rows of inputs to be processed concurrently
        """
        self._graph_repr = graph_repr
        self._parallelism = parallelism
        self.on_task_finish_handlers = []

        self._graph_ctx = None

    @property
    def graph_ctx(self):
        return self._graph_ctx

    def build(self):
        """
        Create GraphContexts and set up input iterators.
        """
        for g in self._graph_ctxs:
            g.on_task_finish_handlers.append(self.on_task_finish_handlers)
        raise NotImplementedError

    def run(self, inputs: list) -> DFIterator:
        """
        The Pipeline's main loop

        Agrs:
            inputs: the input data, organized as a list of DataFrame, feeding
                to the Pipeline.
        """

        # while we still have pipeline inputs:
        #     input = inputs.next()
        #     for g in graph contexts:
        #         if g.is_idle:
        #             g.start_op.inputs = input
        #             break
        #     if all graphs contexts are busy:
        #         wait for notification from _notify_run_loop

        raise NotImplementedError

    def on_start(self, handler: Callable[[], None]) -> None:
        """
        Set a custom handler that called before the execution of the graph.
        """
        self._on_start_handler = handler
        raise NotImplementedError

    def on_finish(self, handler: Callable[[], None]) -> None:
        """
        Set a custom handler that called after the execution of the graph.
        """
        self._on_finish_handler = handler
        raise NotImplementedError

    def _organize_outputs(self, graph_ctx: GraphContext):
        """
        on_finish handler passing to GraphContext. The handler will organize the
        GraphContext's output into Pipeline's outputs.
        """
        raise NotImplementedError

    def _notify_run_loop(self, graph_ctx: GraphContext):
        """
        on_finish handler passing to GraphContext. The handler will notify the run loop
        that a GraphContext is in idle state.
        """
        raise NotImplementedError
