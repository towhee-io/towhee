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
import logging

logger = logging.getLogger()


class ToTritonModel(ABC):
    '''
    Convert a towhee operator to tirton models.
    '''
    def __init__(self, op_meta, model_dir, op_dir):
        self._op_meta = op_meta
        self._model_dir = model_dir
        self._op_dir = op_dir

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


class PythonTritonModel(ToTritonModel):
    def __init__(self, op_meta, model_dir, op_dir):
        super().__init__(op_meta, model_dir)

    def prepare_config(self):
        pass

    def prepare_model(self):
        pass


class TensorRTTritonModel(ToTritonModel):
    def __init__(self, op_meta, model_dir, op_dir):
        super().__init__(op_meta, model_dir)
