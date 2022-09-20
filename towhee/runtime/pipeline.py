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



from towhee.runtime.col_storage import ColStorage
from towhee.runtime.graph import Graph


class DataCollection:
    '''Mock class
    '''
    pass


class Pipeline:
    """
    The runtime pipeline context, include graph context, all dataframes.

    Args:
        graph_repr:
            The graph representation either as a YAML-formatted string, or directly
            as an instance of `GraphRepr`.
    """    
    def __init__(self, graph_repr):
        self._graph_repr = graph_repr

    def __call__(self, *inputs) -> 'DataCollection':
        cols = ColStorage(self._graph_repr.input_schema)
        cols.put(inputs)

        graph = Graph(self._graph_repr.graph)
        output = graph(cols)
        return DataCollection(output)
