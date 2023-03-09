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

from towhee.utils.lazy_import import LazyImport


graph_visualizer = LazyImport('GraphVisualizer', globals(), 'towhee.tools.graph_visualizer')
data_visualizer = LazyImport('PipeVisualizer', globals(), 'towhee.tools.data_visualizer')


def show_graph(dag_repr):
    gv = graph_visualizer.GraphVisualizer(dag_repr)
    gv.show()


def show_data(dag_repr, node_queues):
    pv = data_visualizer.PipeVisualizer(dag_repr, node_queues)
    pv.show()


def get_data_visualizer(dag_repr, node_queues):
    return data_visualizer.PipeVisualizer(dag_repr, node_queues)
