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
import logging

from .util import create_modelconfig, to_triton_schema

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


class PyOpToTriton:
    '''
    PyOp to triton model.
    '''
    def __init__(self, op, model_root, model_name):
        self._op = op
        self._inputs = to_triton_schema(self._op.metainfo['input_schema'], 'INPUT')
        self._outputs = to_triton_schema(self._op.metainfo['output_schema'], 'OUTPUT')        

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs                

    def to_triton(self):
        pass


class ProcessToTriton:
    '''
    Preprocess and Postprocess to triton model.
    '''
    def __init__(self, processor, model_root, model_name, process_type):
        self._processer = processor
        self._trtion_files = TritonFiles(model_root, model_name)
        self._model_name = model_name
        self._processer_type = process_type
        self._process_file = inspect.getmodule(self._processer).__file__
        self._inputs = to_triton_schema(self._processer.metainfo['input_schema'], 'INPUT')
        self._outputs = to_triton_schema(self._processer.metainfo['output_schema'], 'OUTPUT')

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def _prepare_config(self):
        '''
        example of input_schema & output_schema:
            [(np.float32, (-1, -1, 3), (int, ()))]
        '''
        config_str = create_modelconfig(
            self._model_name,
            128,
            self._inputs,
            self._outputs
        )
        with open(self._triton_files.config_file, 'wt') as f:
            f.write(config_str)

    def _preprocess(self):
        # create model.py
        # create pickle file
        with open(self._triton_files.preprocess_pickel, 'wb') as f:
            pickle.dump(self._processer, f)

    def _postprocess(self):
        # create model.py
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


class NNOpToTriton:
    '''
    NNOp to triton model.
    
    Convert model to trt, torchscript or onnx.
    '''
    def __init__(self, op, model_root, model_name):
        self._op = op
        self._trtion_files = TritonFiles(model_root, model_name)
        self._inputs = to_triton_schema(self._op.metainfo['input_schema'], 'INPUT')
        self._outputs = to_triton_schema(self._op.metainfo['output_schema'], 'OUTPUT')

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs        

    def to_triton(self):
        return False
