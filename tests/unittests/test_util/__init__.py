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
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path


GRAPH_TEST_YAML = Path(__file__).parent / 'test_graph/test_graph.yaml'
GRAPH_TEST_ISO_DF_YAML = Path(__file__).parent / 'test_isolated_dataframe/test_isolated_dataframe.yaml'
GRAPH_TEST_ISO_OP_YAML = Path(__file__).parent / 'test_isolated_operator/test_isolated_operator.yaml'
GRAPH_TEST_LOOP_YAML = Path(__file__).parent / 'test_loop/test_loop.yaml'
SIMPLE_PIPELINE_YAML = Path(__file__).parent / 'simple_pipeline/simple_pipeline.yaml'
FLATMAP_PIPELINE_YAML = Path(__file__).parent / 'test_flatmap/flatmap_pipeline.yaml'

