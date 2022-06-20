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
import inspect
import pickle
from abc import ABC
import logging

from .util import create_modelconfig
from serve.triton.triton_config_builder import TritonModelConfigBuilder
from serve.triton.python_model_builder import gen_model_from_op, gen_model_from_pickled_callable

logger = logging.getLogger()


class TritonFiles:
    '''
    File path name.
    '''
    def __init__(self, root, model_name):
        self._root = Path(root) / model_name

    @property
    def root(self):
        return self._root

    @property
    def config_file(self):
        return self._root / 'config.pbtxt'

    @property
    def model_path(self):
        return self._root / '1'

    @property
    def python_model_file(self):
        return self.model_path / 'model.py'

    @property
    def trt_model_file(self):
        return self.model_path / 'model.plan'

    @property
    def onnx_model_file(self):
        return self.model_path / 'model.onnx'

    @property    
    def postprocess_pickel(self):
        return self.model_path / 'postprocess'

    @property
    def preprocess_pickel(self):
        return self.model_path / 'preprocess'


class ToTriton(ABC):
    '''
    ToTriton Base.
    '''
    def __init__(self, obj, model_root, model_name):
        self._obj = obj
        self._model_name = model_name
        self._triton_files = TritonFiles(model_root, model_name)
        self._inputs = TritonModelConfigBuilder.get_input_schema(self._obj.metainfo['input_schema'])
        self._outputs = TritonModelConfigBuilder.get_output_schema(self._obj.metainfo['output_schema'])

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def _create_model_dir(self):
        self._triton_files.root.mkdir(parents=True, exist_ok=True)
        self._triton_files.model_path.mkdir(parents=True, exist_ok=True)
        return True

    def _prepare_model(self):
        return True

    def _prepare_config(self):
        config_str = create_modelconfig(
            self._model_name,
            128,
            self._inputs,
            self._outputs
        )
        with open(self._triton_files.config_file, 'wt', encoding='utf-8') as f:
            f.write(config_str)
        return True

    def to_triton(self):
        if self._create_model_dir() and self._prepare_config() and self._prepare_model():
            return True
        return False


class PyOpToTriton(ToTriton):
    '''
    PyOp to triton model.
    '''
    def __init__(self, op, model_root, model_name,
                 op_hub, op_name, init_args):
        super().__init__(op, model_root, model_name)
        self._op_hub = op_hub
        self._op_name = op_name
        self._init_args = init_args

    def _prepare_model(self):
        gen_model_from_op(self._triton_files.python_model_file,
                          self._op_hub,
                          self._op_name,
                          self._init_args,
                          self._obj.metainfo['input_schema'],
                          self._obj.metainfo['output_schema']
                          )
        return True


class ProcessToTriton(ToTriton):
    '''
    Preprocess and Postprocess to triton model.
    '''
    def __init__(self, processor, model_root, model_name, process_type):
        super().__init__(processor, model_root, model_name)
        self._processer_type = process_type
        self._process_file = inspect.getmodule(self._processer).__file__

    def _preprocess(self):
        gen_model_from_pickled_callable(self._triton_files.python_model_file,
                                        'Preprocess',
                                        self._process_file,
                                        self._triton_files.self._triton_files.preprocess_pickle,
                                        self.inputs,
                                        self.outputs
                                        )
        # create pickle file
        with open(self._triton_files.preprocess_pickle, 'wb') as f:
            pickle.dump(self._processer, f)

    def _postprocess(self):
        # create model.py
        gen_model_from_pickled_callable(self._triton_files.python_model_file,
                                        'Postprocess',
                                        self._process_file,
                                        self._triton_files.self._triton_files.postprocess_pickle,
                                        self.inputs,
                                        self.outputs
                                        )
        # create pickle file
        with open(self._triton_files.postprocess_pickel, 'wb') as f:
            pickle.dump(self._processer, f)

    def _prepare_model(self):
        if self._processer_type == 'preprocess':
            self._preprocess()
        else:
            self._postprocess()

    def to_triton(self):
        if self._prepare_config() and self._prepare_model():
            return True
        return False


class NNOpToTriton(ToTriton):
    '''
    NNOp to triton model.

    Convert model to trt, torchscript or onnx.
    '''
    def to_triton(self):
        if not self._prepare_config():
            pass
        return False
