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
from towhee.errors import NoSchedulerError, EmptyInputError
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
        self._graph_count = 0

        if isinstance(graph_repr, str):
            self._graph_repr = GraphRepr.from_yaml(graph_repr)
        else:
            self._graph_repr = graph_repr

    @property
    def pipeline_type(self) -> int:
        return self._graph_repr.graph_type

    @property
    def parallelism(self) -> int:
        return self._parallelism

    @property
    def graph_repr(self) -> GraphRepr:
        return self._graph_repr

    @parallelism.setter
    def parallelism(self, val):
        if isinstance(val, int) and val > 0:
            self._parallelism = val
        else:
            raise ValueError('Parallelism value must be a positive integer')

    def register(self, scheduler: TaskScheduler):
        self._scheduler = scheduler

    def __repr__(self) -> str:
        if self._graph_repr._ir is not None:
            return self._graph_repr._ir
        else:
            return object.__repr__(self)

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
            raise NoSchedulerError('Pipeline not registered to a Scheduler.')

        assert self._parallelism == 1
        graph_ctx = GraphContext(self._graph_count, self._graph_repr)
        self._graph_count += 1
        self._scheduler.register(graph_ctx)
        input_data = inputs.get(0, 1)
        if input_data is None:
            raise EmptyInputError('Input data is empty')
        graph_ctx(input_data[0])
        graph_ctx.join()
        return graph_ctx.result()
