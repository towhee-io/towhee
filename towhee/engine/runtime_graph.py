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


from typing import List
import threading

from towhee.dag.graph_repr import GraphRepr
from towhee.engine.operator_pool import OperatorPool
from towhee.variable import VariableSet
from towhee.engine.task_queue import TaskQueue


class GraphContext:
    """Subtask runtime global variable, operators can access to get or set data
    """


class RuntimeGraph:
    """Per subtask create one runtime graph
    """
    def __init__(self, graph: GraphRepr) -> None:
        self._graph = graph
        self._graph_context = GraphContext()
        self._operator_pool = OperatorPool()
        self._task_queue = TaskQueue()

    def build(self):
        """create graph nodes and all variable sets
        """
        raise NotImplementedError

    def start(self):
        """Put nodes to task queue
        """
        raise NotImplementedError


class Node:
    def __init__(self, op, inputs: VariableSet, outputs: List[VariableSet]):
        self._op = op
        self._inputs = inputs
        self._outputs = outputs
    
    def _read_inputs(self) -> dict:
        """iterator all inputs, get op input params
        """
        raise NotImplementedError

    def run(self):
        """Reads all inputs, call op.__call__(), and set result to all outputs

            for data in self._read_inputs():
                result = self._op(**data)
                for output in self._outputs:
                    output.add(result)
        """
        raise NotImplementedError

