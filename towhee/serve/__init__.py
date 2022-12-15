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

# pylint: disable = dangerous-default-value
import copy

from towhee.utils.log import engine_log

from .triton.pipeline_client import Client as triton_client
from .triton.pipeline_builder import Builder as TritonModelBuilder
from .triton.docker_image_builder_v2 import DockerImageBuilder


def build_docker_image_v2(
    dc_pipeline: 'towhee.RuntimePipeline', image_name: str,  cuda_version: str, format_priority: list = ['onnx'], inference_server: str = 'triton'
):
    server_config = {'format_priority' : format_priority}
    if inference_server == 'triton':
        DockerImageBuilder(dc_pipeline, image_name, server_config, cuda_version).build()
    else:
        engine_log.error('Unknown server type: %s.', inference_server)
        return False


def build_pipeline_model(dc_pipeline: 'towhee.RuntimePipeline', model_root: str, format_priority: list, server: str = 'triton'):
    if server == 'triton':
        dag_repr = copy.deepcopy(dc_pipeline.dag_repr)
        server_conf = {
            'format_priority': format_priority
        }
        builder = TritonModelBuilder(dag_repr, model_root, server_conf)
    else:
        engine_log.error('Unknown server type: %s.', server)
        builder = None

    if builder is not None:
        return builder.build()
    return False
