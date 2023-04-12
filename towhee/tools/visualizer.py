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
import json
from typing import Union, List, Dict, Any

from towhee.runtime.time_profiler import TimeProfiler
from towhee.tools.profilers import PerformanceProfiler
from towhee.datacollection import DataCollection
from towhee.utils.lazy_import import LazyImport
from towhee.utils.log import engine_log


graph_visualizer = LazyImport('graph_visualizer', globals(), 'towhee.tools.graph_visualizer')
data_visualizer = LazyImport('data_visualizer', globals(), 'towhee.tools.data_visualizer')


def show_graph(pipeline):
    gv = graph_visualizer.GraphVisualizer(pipeline.dag_repr)
    gv.show()


class Visualizer:
    """
    Visualize the debug information.
    """
    def __init__(
        self,
        result: Union['DataQueue', List[Any]]=None,
        time_profiler: List[Any]=None,
        data_queues: List[Dict[str, Any]]=None,
        nodes: Dict[str, Any]=None,
        trace_nodes: List[str]=None
    ):
        self._result = result
        self._time_profiler = time_profiler
        self._data_queues = data_queues
        self._trace_nodes = trace_nodes
        self._nodes = nodes
        self._node_collection = [self._get_collection(i) for i in self._get_node_queues()] if self._data_queues else None
        self._profiler = None
        self._tracer = None

    def _get_node_queues(self):
        """
        Get node queue with given graph data queues.
        """
        node_queues = []
        for data_queue in self._data_queues:
            node_queue = {}
            for node in self._nodes.values():
                if node['name'] not in self._trace_nodes:
                    continue
                node_queue[node['name']] = {}
                node_queue[node['name']]['in'] = [data_queue[edge] for edge in node['inputs']]
                node_queue[node['name']]['out'] = [data_queue[edge] for edge in node['outputs']]
                node_queue[node['name']]['op_input'] = node['op_input']
                node_queue[node['name']]['next'] = [self._nodes[i]['name'] for i in node['next_nodes']]
            self._set_previous(node_queue)
            node_queues.append(node_queue)

        return node_queues

    def _set_previous(self, node_queue):
        for node in self._nodes.values():
            for i in node['next_nodes']:
                next_node = self._nodes[i]['name']
                if next_node not in self._trace_nodes:
                    continue
                if 'previous' not in node_queue[next_node]:
                    node_queue[next_node]['previous'] = [node['name']]
                else:
                    node_queue[next_node]['previous'].append(node['name'])

    @staticmethod
    def _get_collection(node_info):

        def _to_collection(x):
            for idx, q in enumerate(x):
                if not q.size:
                    q.reset_size()
                tmp = DataCollection(q)
                q.reset_size()
                x[idx] = tmp

        for v in node_info.values():
            _to_collection(v['in'])
            _to_collection(v['out'])

        return node_info

    @property
    def result(self):
        return self._result

    @property
    def profiler(self):
        if not self._time_profiler:
            w_msg = 'Please set `profiler` to `True` when debug, there is nothing to report.'
            engine_log.warning(w_msg)
            return None
        if not self._profiler:
            self._profiler = PerformanceProfiler(self._time_profiler, self._nodes)

        return self._profiler

    @property
    def tracer(self):
        if not self._node_collection:
            w_msg = 'Please set `tracer` to `True` when debug, there is nothing to report.'
            engine_log.warning(w_msg)
            return None
        if not self._tracer:
            self._tracer = data_visualizer.DataVisualizer(self._nodes, self._node_collection)
        return self._tracer

    @property
    def time_profiler(self):
        return self._time_profiler

    @property
    def node_collection(self):
        return self._node_collection

    @property
    def nodes(self):
        return self._nodes

    def _to_dict(self):
        info = {}
        if self._result:
            info['result'] = self._result
        if self._time_profiler:
            info['time_record'] = [i.time_record for i in self._time_profiler]
        if self._nodes:
            info['nodes'] = self._nodes

        return info

    def to_json(self, **kws):
        return json.dumps(self._to_dict(), **kws)

    @staticmethod
    def from_json(info):
        info_dict = json.loads(info)
        return Visualizer(
            result=info_dict.get('result'),
            time_profiler=[TimeProfiler(enable=True, time_record=i) for i in info_dict.get('time_record')],
            nodes=info_dict.get('nodes')
        )
