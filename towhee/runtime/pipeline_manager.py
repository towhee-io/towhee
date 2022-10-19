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

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from towhee.utils.log import engine_log
from .operator_manager import OperatorPool
from .data_queue import DataQueue
from .dag_repr import DAGRepr
from .nodes import create_node, NodeStatus
from .node_repr import NodeRepr


class Graph:
    """
    Graph.

    Args:
        nodes(`Dict[str, NodeRepr]`): The pipeline nodes from DAGRepr.nodes.
        edges(Dict[str, List]`: The pipeline edges from DAGRepr.edges.
        operator_pool(`OperatorPool`): The operator pool.
    """
    def __init__(self, nodes: Dict[str, NodeRepr], edges: Dict[str, Any], operator_pool: 'OperatorPool', thread_pool: 'ThreadPoolExecutor'):
        self._nodes = nodes
        self._edges = edges
        self._operator_pool = operator_pool
        self._thread_pool = thread_pool
        self._node_runners = None
        self._data_queues = None

    def initialize(self):
        self._node_runners = []
        self._data_queues = dict((name, DataQueue(edge['data'])) for name, edge in self._edges.items())
        for name in DAGRepr.get_top_sort(self._nodes):
            in_queues = [self._data_queues[edge] for edge in self._nodes[name].in_edges]
            out_queues = [self._data_queues[edge] for edge in self._nodes[name].out_edges]
            node = create_node(self._nodes[name], self._operator_pool, in_queues, out_queues)
            if not node.initialize():
                raise RuntimeError(node.err_msg)
            self._node_runners.append(node)

    def result(self) -> any:
        end_edge_num = self._nodes['_output'].out_edges[0]
        res = self._data_queues[end_edge_num]
        for node in self._node_runners:
            if node.status != NodeStatus.FINISHED:
                raise RuntimeError(node.err_msg)
        if res.size != 0:
            return res.get()
        else:
            engine_log.warning('The pipeline runs successfully, but no data return')
            return None

    def __call__(self, *inputs):
        self.initialize()
        self._data_queues[0].put(*inputs)
        self._data_queues[0].seal()
        features = []
        for node in self._node_runners:
            features.append(self._thread_pool.submit(node.process))
        _ = [f.result() for f in features]
        return self.result()

    def release(self):
        del self._data_queues
        del self._node_runners


class PipelineManager:
    """
    Manage the pipeline and runs it as a single instance.


    Args:
        dag_repr(`Dict`): A DAGRepr or Dictionary from the user pipeline.
    """

    def __init__(self, dag_repr: Dict):
        self._dag_repr = DAGRepr.from_dict(dag_repr)
        self._operator_pool = OperatorPool()
        self._thread_pool = ThreadPoolExecutor()

    def preload(self):
        """
        Initialize the nodes.
        """
        graph = Graph(self._dag_repr.nodes, self._dag_repr.edges, self._operator_pool, self._thread_pool)
        graph.initialize()

    def __call__(self, *inputs):
        """
        Output with ordering matching the input `DataQueue`.
        """
        graph = Graph(self._dag_repr.nodes, self._dag_repr.edges, self._operator_pool, self._thread_pool)
        outputs = graph(inputs)
        graph.release()
        return outputs
