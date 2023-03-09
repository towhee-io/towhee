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
from towhee.runtime import register, pipe, ops, accelerate, AutoConfig, AutoPipes


datacollection = LazyImport('datacollection', globals(), 'towhee.datacollection')
server_builder = LazyImport('server_builder', globals(), 'towhee.serve.server_builder')


def build_docker_image_v2(*args, **kwargs):
    """
    Wrapper for lazy import build_docker_image_v2
    """
    return server_builder.build_docker_image_v2(*args, **kwargs)


def build_pipeline_model(*args, **kwargs):
    """
    Wrapper for lazy import build_pipeline_model
    """
    return server_builder.build_pipeline_model(*args, **kwargs)


def DataCollection(*args, **kwargs):  # pylint: disable=invalid-name
    """
    Wrapper for lazy import DataCollection
    """
    return datacollection.DataCollection(*args, **kwargs)


__all__ = [
    'register',
    'pipe',
    'ops',
    'accelerate',
    'DataCollection',
    'build_docker_image_v2',
    'build_pipeline_model',
    'AutoConfig',
    'AutoPipes'
]
