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
from towhee.utils.lazy_import import LazyImport


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
        tracer: Dict[str, Any]=None,
        nodes:Dict[str, Any]=None,
    ):
        self._result = result
        self._time_profiler = time_profiler
        self._tracer = tracer
        self._nodes = nodes
        self._profiler = None

    @property
    def result(self):
        return self._result

    @property
    def profiler(self):
        if not self._profiler:
            self._profiler = PerformanceProfiler(self._time_profiler, self._nodes)

        return self._profiler

    @property
    def time_profiler(self):
        return self._time_profiler

    @property
    def nodes(self):
        return self._nodes

    def show_data(self):
        pv = data_visualizer.PipeVisualizer(self._nodes, self._tracer)
        pv.show()

    def get_data_visualizer(self):
        return data_visualizer.PipeVisualizer(self._nodes, self._tracer)

    def _to_dict(self):
        info = {}
        if self._result:
            info['result'] = self._result
        if self._time_profiler:
            info['time_record'] = [i.time_record for i in self._time_profiler]
        if self._tracer:
            info['tracer'] = self._tracer
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
            tracer=info_dict.get('tracer'),
            nodes=info_dict.get('nodes')
        )
