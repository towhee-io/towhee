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


from pathlib import Path


class TritonFiles:
    '''
    File path name.
    '''
    def __init__(self, root: str, model_name: str):
        self._root = Path(root) / model_name

    @property
    def root(self) -> Path:
        return self._root

    @property
    def config_file(self) -> Path:
        return self._root / 'config.pbtxt'

    @property
    def model_path(self) -> Path:
        return self._root / '1'

    @property
    def python_model_file(self) -> Path:
        return self.model_path / 'model.py'

    @property
    def pipe_pickle(self) -> Path:
        return 'pipe.pickle'

    @property
    def pipe_pickle_path(self) -> Path:
        return self.model_path / self.pipe_pickle

    @property
    def trt_model_file(self) -> Path:
        return self.model_path / 'model.plan'

    @property
    def onnx_model_file(self) -> Path:
        return self.model_path / 'model.onnx'
