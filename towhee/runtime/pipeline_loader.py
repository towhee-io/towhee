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


import threading
import importlib

from towhee.pipelines import get_builtin_pipe_file


PIPELINE_NAMESPACE = 'towhee.pipeline'


class PipelineLoader:
    _lock = threading.Lock()
    
    @staticmethod
    def model_name(name):
        return PIPELINE_NAMESPACE + '.' + name

    @staticmethod
    def _load_pipeline_from_file(name: str, file_path: str):
        modname = PipelineLoader.model_name(name)
        spec = importlib.util.spec_from_file_location(modname, file_path.resolve())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    @staticmethod
    def load_pipeline(name):
        with PipelineLoader._lock:
            file_path = get_builtin_pipe_file(name)
            if file_path is not None:
                PipelineLoader._load_pipeline_from_file(name, file_path)
            else:
                # TODO load pipeline from hub
                pass
            
