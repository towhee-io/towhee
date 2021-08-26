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
from towhee.data_source import DataSource
from towhee.engine.runtime_graph import RuntimeGraph


class LocalEngine:
    """
    """
    def __init__(self, graph: GraphRepr, ds: DataSource) -> None:
        self._graph = graph
        self._ds = ds

    def _process(self):
        """Control Parallelism of engine,
        """
        raise NotImplementedError

    def run(self):
        """Read from data source, create & and run graph 

           for data in self._ds:
               # input_variable_set = VariableSet.add(data)
               runtime_graph = RuntimeGraph(self._graph, data)
               runtime_graph.build()
               self._process(runtime_graph)

        """
        raise NotImplementedError
