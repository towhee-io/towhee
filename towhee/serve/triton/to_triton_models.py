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

from typing import Dict
from pathlib import Path
import inspect
import pickle
from abc import ABC
import logging

from towhee.serve.triton.triton_config_builder import TritonModelConfigBuilder, create_modelconfig, EnsembleConfigBuilder
from towhee.serve.triton.python_model_builder import gen_model_from_op, gen_model_from_pickled_callable

logger = logging.getLogger()


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
    def trt_model_file(self) -> Path:
        return self.model_path / 'model.plan'

    @property
    def onnx_model_file(self) -> Path:
        return self.model_path / 'model.onnx'

    @property
    def preprocess_pickle(self) -> Path:
        return 'preprocess.pickle'

    @property
    def postprocess_pickle(self) -> Path:
        return 'postprocess.pickle'

    @property
    def postprocess_pickle_path(self) -> Path:
        return self.model_path / self.postprocess_pickle

    @property
    def preprocess_pickle_path(self) -> Path:
        return self.model_path / self.preprocess_pickle


class ToTriton(ABC):
    '''
    ToTriton Base.
    '''
    def __init__(self, obj: 'Operator', model_root: str, model_name: str, os_config: Dict):
        self._obj = obj
        self._model_name = model_name
        self._os_config = os_config
        self._triton_files = TritonFiles(model_root, self._model_name)
        self._inputs = TritonModelConfigBuilder.get_input_schema(self._obj.input_schema())
        self._outputs = TritonModelConfigBuilder.get_output_schema(self._obj.output_schema())
        self._backend = 'python'

    @property
    def inputs(self) -> Dict:
        return self._inputs

    @property
    def outputs(self) -> Dict:
        return self._outputs

    def _create_model_dir(self) -> bool:
        self._triton_files.root.mkdir(parents=True, exist_ok=True)
        self._triton_files.model_path.mkdir(parents=True, exist_ok=True)
        return True

    def _prepare_model(self) -> bool:
        return True

    def _prepare_config(self) -> bool:
        device_ids = self._os_config.get('device_ids')
        instance_count = self._os_config.get('instance_count', 1)
        config_str = create_modelconfig(
            self._model_name,
            0,
            self._inputs,
            self._outputs,
            self._backend,
            False,
            None,
            None,
            instance_count,
            device_ids
        )
        with open(self._triton_files.config_file, 'wt', encoding='utf-8') as f:
            f.write(config_str)
            return True

        with open(self._triton_files.config_file, 'wt', encoding='utf-8') as f:
            f.write(config_str)
        return True

    def to_triton(self) -> bool:
        if self._create_model_dir() and self._prepare_model() and self._prepare_config():
            return True
        return False


class PyOpToTriton(ToTriton):
    '''
    PyOp to triton model.
    '''
    def __init__(self, op, model_root, model_name,
                 op_hub, op_name, init_args, os_config):
        super().__init__(op, model_root, model_name, os_config)
        self._op_hub = op_hub
        self._op_name = op_name
        self._init_args = init_args

    def _prepare_model(self):
        gen_model_from_op(self._triton_files.python_model_file,
                          self._op_hub,
                          self._op_name,
                          self._init_args,
                          self._obj.input_schema(),
                          self._obj.output_schema()
                          )
        return True


class PreprocessToTriton(ToTriton):
    '''
    Preprocess to triton model.
    '''
    def __init__(self, op, model_root, model_name, os_config):
        super().__init__(op.preprocess, model_root, model_name, os_config)
        op_module_info = inspect.getmodule(op)
        self._init_file = Path(op_module_info.__file__).parent / '__init__.py'
        self._module_name = '.'.join(op_module_info.__name__.split('.')[:-1])

    def _prepare_model(self):
        gen_model_from_pickled_callable(str(self._triton_files.python_model_file),
                                        self._module_name,
                                        str(self._init_file),
                                        str(self._triton_files.preprocess_pickle),
                                        self._obj.input_schema(),
                                        self._obj.output_schema()
                                        )
        # create pickle file
        with open(self._triton_files.preprocess_pickle_path, 'wb') as f:
            pickle.dump(self._obj, f)
        return True


class PostprocessToTriton(ToTriton):
    '''
    Preprocess and Postprocess to triton model.
    '''
    def __init__(self, op, model_root, model_name, os_config):
        super().__init__(op.postprocess, model_root, model_name, os_config)
        op_module_info = inspect.getmodule(op)
        self._init_file = Path(op_module_info.__file__).parent / '__init__.py'
        self._module_name = '.'.join(op_module_info.__name__.split('.')[:-1])

    def _prepare_model(self):
        gen_model_from_pickled_callable(str(self._triton_files.python_model_file),
                                        self._module_name,
                                        str(self._init_file),
                                        str(self._triton_files.postprocess_pickle),
                                        self._obj.input_schema(),
                                        self._obj.output_schema()
                                        )
        # create pickle file
        with open(self._triton_files.postprocess_pickle_path, 'wb') as f:
            pickle.dump(self._obj, f)
        return True


class ModelToTriton (ToTriton):
    '''
    NNOp to triton model.

    Convert model to trt, torchscript or onnx.
    '''
    def __init__(self, op, model_root, model_name, model_format_priority, os_config):
        super().__init__(op.model, model_root, model_name, os_config)
        self._model_format_priority = model_format_priority

    def _prepare_config(self) -> bool:
        '''
        All model open dynamic_batching.
        After towhee.dc supports config the batch info, we use the config.
        example of input and output:
        {
            INPUT0': ('TYPE_INT8', [-1, -1, 3]),
            INPUT1': ('TYPE_FP32', [-1, -1, 3])
        }
        '''
        dynamic_batching = self._os_config.get('dynamic_batching', {})
        device_ids = self._os_config.get('device_ids')
        instance_count = self._os_config.get('instance_count', 1)
        if dynamic_batching:
            # remove the batch dim
            inputs = {}
            for k, v in self._inputs.items():
                inputs[k] = (v[0], v[1][1:])
                outputs = {}
            for k, v in self._outputs.items():
                outputs[k] = (v[0], v[1][1:])
            max_batch_size = dynamic_batching.get('max_batch_size', 1)
            enable_dynamic_batching = True
            preferred_batch_size = dynamic_batching.get('preferred_batch_size', None)
            preferred_max_queue_delay_microseconds = dynamic_batching.get('preferred_max_queue_delay_microseconds', None)

            config_str = create_modelconfig(
                self._model_name,
                max_batch_size,
                inputs,
                outputs,
                self._backend,
                enable_dynamic_batching,
                preferred_batch_size,
                preferred_max_queue_delay_microseconds,
                instance_count,
                device_ids
            )
        else:
            config_str = create_modelconfig(
                self._model_name,
                0,
                self._inputs,
                self._outputs,
                self._backend,
                False,
                None,
                None,
                instance_count,
                device_ids
            )
        with open(self._triton_files.config_file, 'wt', encoding='utf-8') as f:
            f.write(config_str)
            return True

    def _prepare_model(self):
        succ = False
        for fmt in self._model_format_priority:
            if fmt in self._obj.supported_formats:
                if fmt == 'onnx':
                    succ = self._obj.save_model(fmt, self._triton_files.onnx_model_file)
                    self._backend = 'onnxruntime'
                elif fmt == 'tensorrt':
                    succ = self._obj.save_model(fmt, self._triton_files.trt_model_file)
                    self._backend = 'tensorrt'
                else:
                    logger.error('Unkown optimize %s', fmt)
                    continue
        return succ


class EnsembleToTriton:
    '''
    Create triton ensemble config
    '''
    def __init__(self, dag, model_root, model_name, batch_size):
        self._dag = dag
        self._model_name = model_name
        self._triton_files = TritonFiles(model_root, self._model_name)
        self._batch_size = batch_size

    def to_triton(self):
        self._triton_files.root.mkdir(parents=True, exist_ok=True)
        self._triton_files.model_path.mkdir(parents=True, exist_ok=True)
        config_str = EnsembleConfigBuilder(self._dag, self._model_name, self._batch_size).gen_config()
        with open(self._triton_files.config_file, 'wt', encoding='utf-8') as f:
            f.write(config_str)
        return True
