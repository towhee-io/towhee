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
from .performance_profiler import PerformanceProfiler, TimeProfiler, Event
from .constants import TracerConst


class Graph:
    """
    Graph.

    Args:
        nodes(`Dict[str, NodeRepr]`): The pipeline nodes from DAGRepr.nodes.
        edges(`Dict[str, Any]`): The pipeline edges from DAGRepr.edges.
        operator_pool(`OperatorPool`): The operator pool.
        thread_pool(`OperatorPool`): The ThreadPoolExecutor.
    """
    def __init__(self,
                 nodes: Dict[str, NodeRepr],
                 edges: Dict[str, Any],
                 operator_pool: 'OperatorPool',
                 thread_pool: 'ThreadPoolExecutor',
                 time_profiler: 'TimeProfiler' = None):
        self._nodes = nodes
        self._edges = edges
        self._operator_pool = operator_pool
        self._thread_pool = thread_pool
        self._time_profiler = time_profiler
        self._node_runners = None
        self._data_queues = None
        self.initialize()

    def initialize(self):
        self._node_runners = []
        self._data_queues = dict((name, DataQueue(edge['data'])) for name, edge in self._edges.items())
        for name in self._nodes:
            in_queues = [self._data_queues[edge] for edge in self._nodes[name].in_edges]
            out_queues = [self._data_queues[edge] for edge in self._nodes[name].out_edges]
            node = create_node(self._nodes[name], self._operator_pool, in_queues, out_queues, self._time_profiler)
            if not node.initialize():
                raise RuntimeError(node.err_msg)
            self._node_runners.append(node)

    def result(self) -> any:
        errs = ''
        for node in self._node_runners:
            if node.status != NodeStatus.FINISHED:
                if node.status ==NodeStatus.FAILED:
                    errs += node.err_msg + '\n'
        if errs:
            raise RuntimeError(errs)
        end_edge_num = self._nodes['_output'].out_edges[0]
        res = self._data_queues[end_edge_num]
        return res

    def __call__(self, *inputs):
        self._data_queues[0].put(*inputs)
        self._data_queues[0].seal()
        features = []
        for node in self._node_runners:
            features.append(self._thread_pool.submit(node.process))

        _ = [f.result() for f in features]
        return self.result()


class RuntimePipeline:
    """
    Manage the pipeline and runs it as a single instance.


    Args:
        dag_dict(`Dict`): The DAG Dictionary from the user pipeline.
        max_workers(`int`): The maximum number of threads.
    """

    def __init__(self, dag_dict: Dict, max_workers: int = None, config: dict = None):
        self._dag_repr = DAGRepr.from_dict(dag_dict)
        self._operator_pool = OperatorPool()
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._time_profiler_list = []
        self._config = config

    def preload(self):
        """
        Preload the operators.
        """
        _ = Graph(self._dag_repr.nodes, self._dag_repr.edges, self._operator_pool, self._thread_pool)

    def __call__(self, *inputs):
        """
        Output with ordering matching the input `DataQueue`.
        """
        time_profiler = TimeProfiler()
        if self._config is not None and TracerConst.name in self._config and self._config[TracerConst.name]:
            time_profiler.enable()

        time_profiler.record(Event.pipe_name, Event.pipe_in, inputs)
        graph = Graph(self._dag_repr.nodes, self._dag_repr.edges, self._operator_pool, self._thread_pool, time_profiler)
        outputs = graph(inputs)
        time_profiler.record(Event.pipe_name, Event.pipe_out)

        self._time_profiler_list.append(time_profiler)
        return outputs

    def profiler(self):
        """
        Report the performance results after running the pipeline, and please note that you need to set tracer to True when you declare a pipeline.
        """
        records = []
        for time_profiler in self._time_profiler_list:
            records += time_profiler.time_record
        if len(records) == 0:
            engine_log.warning('Please set tracer to True or you need to run it first, there is nothing to report.')
            return None
        performance_profiler = PerformanceProfiler(records, self._dag_repr)
        return performance_profiler

    def reset_tracer(self):
        """
        Reset the tracer, reset the record to None.
        """
        self._time_profiler_list = []
