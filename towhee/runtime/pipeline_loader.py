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


import pathlib
import hashlib
import threading
import importlib


from towhee.pipelines import get_builtin_pipe_file
from towhee.hub import get_pipeline


PIPELINE_NAMESPACE = 'towhee.pipeline'


class PipelineLoader:
    """
    Load Predefined Pipelines
    """
    _lock = threading.Lock()

    @staticmethod
    def module_name(name):
        name = name.replace('/', '.')
        return PIPELINE_NAMESPACE + '.' + name

    @staticmethod
    def _load_pipeline_from_file(name: str, file_path: 'Path'):
        modname = PipelineLoader.module_name(name)
        spec = importlib.util.spec_from_file_location(modname, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    @staticmethod
    def _load_builtins(name: str) -> bool:
        file_path = get_builtin_pipe_file(name)
        if file_path is None:
            return False
        PipelineLoader._load_pipeline_from_file(name, file_path)
        return True

    @staticmethod
    def load_pipeline(name: str, tag: str = 'main', latest: bool = False):
        with PipelineLoader._lock:
            file_path = pathlib.Path(name)
            if file_path.is_file():
                new_name = hashlib.sha256(name.encode('utf-8')).hexdigest()
                PipelineLoader._load_pipeline_from_file(new_name, file_path)
            else:
                if not PipelineLoader._load_builtins(name):
                    path = get_pipeline(name, tag, latest)
                    pipe_name = name.replace('-', '_')
                    file_path = path / (pipe_name.split('/')[-1] + '.py')
                    new_name = pipe_name + '.' + tag
                    PipelineLoader._load_pipeline_from_file(new_name, file_path)
