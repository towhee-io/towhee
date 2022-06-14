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

from abc import ABC
from path import Pathlib
import logging

logger = logging.getLogger()


class TritonFiles:
    def __init__(self, root, model_name):
        self._root = Pathlib(root) / model_name

    @property
    def root(self):
        return self._root

    @property
    def config_file(self):
        return self._root / 'config.pbtxt'

    @property
    def model_path(self):
        return self._root / 1

    @property
    def python_model_file(self):
        return self.model_path / 'model.py'

    @property
    def trt_model_file(self):
        return self.model_path / 'model.plan'

    @property
    def onnx_model_file(self):
        return self.model_path / 'model.onnx'


class ToTritonModel(ABC):
    '''
    Convert a towhee operator to tirton models.
    '''
    def __init__(self, op_meta, model_dir, op_dir):
        self._op_meta = op_meta
        self._op_dir = op_dir
        self._triton_files = TritonFiles(model_dir, op_meta.model_name)

    def _prepare_config(self):
        pass

    def _prepare_model(self):
        pass

    def _create_workspace(self):
        pass

    def to_triton(self):
        if not self._create_workspace():
            return False

        if not self._prepare_config():
            return False

        if not self._prepare_model():
            return False
        return True


class EnsembleTritionModel:
    def __init__(self, dag):
        pass

    def _prepare_config(self):
        pass

    def _prepare_model(self):
        return True


class PythonTritonModel(ToTritonModel):
    def __init__(self, op_meta, model_dir, op_dir):
        super().__init__(op_meta, model_dir)

    def _prepare_config(self):
        pass

    def _prepare_model(self):
        pass


class TorchTensorRTTritonModel(ToTritonModel):
    def __init__(self, op_meta, model_dir, op_dir):
        super().__init__(op_meta, model_dir)

    def _prepare_config(self):
        pass

    def _prepare_model(self):
        pass
