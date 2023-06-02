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

import re
from typing import Dict, Any, Union, Tuple, List
from concurrent.futures import ThreadPoolExecutor

from towhee.tools import visualizers
from towhee.utils.log import engine_log
from .operator_manager import OperatorPool
from .data_queue import DataQueue
from .dag_repr import DAGRepr
from .nodes import create_node, NodeStatus
from .node_repr import NodeRepr
from .time_profiler import TimeProfiler, Event


class _GraphResult:
    def __init__(self, graph: '_Graph'):
        self._graph = graph

    def result(self):
        ret = self._graph.result()
        self._graph.release_op()
        del self._graph
        return ret


class _Graph:
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
                 time_profiler: 'TimeProfiler' = None,
                 trace_edges: list = None):
        self._nodes = nodes
        self._edges = edges
        self._operator_pool = operator_pool
        self._thread_pool = thread_pool
        self._time_profiler = time_profiler
        self._trace_edges = trace_edges
        self._node_runners = None
        self._data_queues = None
        self.features = None
        self._time_profiler.record(Event.pipe_name, Event.pipe_in)
        self._initialize()
        self._input_queue = self._data_queues[0]

    def _initialize(self):
        self._node_runners = []
        self._data_queues = dict(
            (
                name,
                DataQueue(edge['data'], keep_data=(self._trace_edges and self._trace_edges.get(name, False)))
            ) for name, edge in self._edges.items()
        )
        for name in self._nodes:
            in_queues = [self._data_queues[edge] for edge in self._nodes[name].in_edges]
            out_queues = [self._data_queues[edge] for edge in self._nodes[name].out_edges]
            node = create_node(self._nodes[name], self._operator_pool, in_queues, out_queues, self._time_profiler)
            if not node.initialize():
                raise RuntimeError(node.err_msg)
            self._node_runners.append(node)

    def result(self) -> any:
        for f in self.features:
            f.result()
        errs = ''
        for node in self._node_runners:
            if node.status != NodeStatus.FINISHED:
                if node.status == NodeStatus.FAILED:
                    errs += node.err_msg + '\n'
        if errs:
            raise RuntimeError(errs)
        end_edge_num = self._nodes['_output'].out_edges[0]
        res = self._data_queues[end_edge_num]
        self.time_profiler.record(Event.pipe_name, Event.pipe_out)
        return res

    def async_call(self, inputs: Union[Tuple, List]):
        self.time_profiler.inputs = inputs
        self._input_queue.put(inputs)
        self._input_queue.seal()
        self.features = []
        for node in self._node_runners:
            self.features.append(self._thread_pool.submit(node.process))
        return _GraphResult(self)

    def release_op(self):
        for node in self._node_runners:
            node.release_op()

    def __call__(self, inputs: Union[Tuple, List]):
        f = self.async_call(inputs)
        return f.result()

    @property
    def time_profiler(self):
        return self._time_profiler

    @property
    def input_col_size(self):
        return self._input_queue.col_size

    @property
    def data_queues(self):
        return self._data_queues


class RuntimePipeline:
    """
    Manage the pipeline and runs it as a single instance.


    Args:
        dag_dict(`Dict`): The DAG Dictionary from the user pipeline.
        max_workers(`int`): The maximum number of threads.
    """

    def __init__(self, dag: Union[Dict, DAGRepr], max_workers: int = None):
        if isinstance(dag, Dict):
            self._dag_repr = DAGRepr.from_dict(dag)
        else:
            self._dag_repr = dag
        self._operator_pool = OperatorPool()
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def preload(self):
        """
        Preload the operators.
        """
        return _Graph(self._dag_repr.nodes, self._dag_repr.edges, self._operator_pool, self._thread_pool, TimeProfiler(False))

    def __call__(self, *inputs):
        """
        Output with ordering matching the input `DataQueue`.
        """
        return self._call(*inputs, profiler=False, tracer=False)[0]

    def batch(self, batch_inputs):
        return self._batch(batch_inputs, profiler=False, tracer=False)[0]

    def flush(self):
        """
        Call the flush interface of ops.
        """
        self._operator_pool.flush()

    def _call(self, *inputs, profiler: bool, tracer: bool, trace_edges: list = None):
        """
        Run pipeline with debug option.
        """
        time_profiler = TimeProfiler(True) if profiler else TimeProfiler(False)
        graph = _Graph(self._dag_repr.nodes, self._dag_repr.edges, self._operator_pool, self._thread_pool, time_profiler, trace_edges)

        return graph(inputs), [graph.time_profiler] if profiler else None, [graph.data_queues] if tracer else None

    def _batch(self, batch_inputs, profiler: bool, tracer: bool, trace_edges: list = None):
        """
        Run batch call with debug option.
        """
        graph_res = []
        time_profilers = []
        data_queues = []
        for inputs in batch_inputs:
            time_profiler = TimeProfiler(False) if time_profilers is None else TimeProfiler(True)
            gh = _Graph(self._dag_repr.nodes, self._dag_repr.edges, self._operator_pool, self._thread_pool, time_profiler, trace_edges)

            if profiler:
                time_profilers.append(gh.time_profiler)
            if tracer:
                data_queues.append(gh.data_queues)
            if gh.input_col_size == 1:
                inputs = (inputs, )
            graph_res.append(gh.async_call(inputs))

        rets = []
        for gf in graph_res:
            ret = gf.result()
            rets.append(ret)
        return rets, time_profilers if time_profilers else None, data_queues if data_queues else None

    @property
    def dag_repr(self):
        return self._dag_repr

    def _get_trace_nodes(self, include, exclude):
        def _find_match(patterns, x):
            return any(re.search(pattern, x) for pattern in patterns)

        include = [include] if isinstance(include, str) else include
        exclude = [exclude] if isinstance(exclude, str) else exclude
        include = [node.name for node in self._dag_repr.nodes.values() if _find_match(include, node.name)] if include else []
        exclude = [node.name for node in self._dag_repr.nodes.values() if _find_match(exclude, node.name)] if exclude else []
        trace_nodes = list(set(include) - set(exclude)) if include \
            else list(set(node.name for node in self._dag_repr.nodes.values()) - set(exclude))

        return trace_nodes

    def _get_trace_edges(self, trace_nodes):
        def _set_false(idx):
            trace_edges[idx] = False

        trace_edges = dict((id, True) for id in self.dag_repr.edges)
        for node in self.dag_repr.nodes.values():
            if node.name not in trace_nodes:
                _ = [_set_false(i) for i in node.out_edges]

        return trace_edges

    def debug(
        self,
        *inputs,
        batch: bool = False,
        profiler: bool = False,
        tracer: bool = False,
        include: Union[List[str], str] = None,
        exclude: Union[List[str], str] = None
    ):
        """
        Run pipeline in debug mode.

        One can record the running time of each operator by setting `profiler` to `True`, or record the data of itermediate nodes
        by setting `tracer` to True. Note that one should at least specify one of `profiler` and `tracer` options to True.
        When debug with `tracer` option, one can specify which nodes to include or exclude.

        Args:
            batch (`bool):
                Whether to run in batch mode.
            profiler (`bool`):
                Whether to record the performance of the pipeline.
            tracer (`bool`):
                Whether to record the data from intermediate nodes.
            include (`Union[List[str], str]`):
                The nodes not to trace.
            exclude (`Union[List[str], str]`):
                The nodes to trace.
        """
        if not profiler and not tracer:
            e_msg = 'You should set at least one of `profiler` or `tracer` to `True` when debug.'
            engine_log.error(e_msg)
            raise ValueError(e_msg)

        trace_nodes = self._get_trace_nodes(include, exclude)
        trace_edges = self._get_trace_edges(trace_nodes)

        time_profilers = [] if profiler else None
        data_queues = [] if tracer else None

        if not batch:
            res, time_profilers, data_queues = self._call(*inputs, profiler=profiler, tracer=tracer, trace_edges=trace_edges)
        else:
            res, time_profilers, data_queues = self._batch(inputs[0], profiler=profiler, tracer=tracer, trace_edges=trace_edges)

        v = visualizers.Visualizer(
            result=res, time_profiler=time_profilers, data_queues=data_queues ,nodes=self._dag_repr.to_dict().get('nodes'), trace_nodes=trace_nodes
        )

        return v
