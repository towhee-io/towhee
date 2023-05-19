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


import traceback
from towhee.utils.log import engine_log
from towhee.utils.onnx_utils import onnx
from towhee.serve.triton import constant
from towhee.serve.triton.triton_config_builder import create_modelconfig
from towhee.serve.triton.triton_files import TritonFiles


class ModelToTriton:
    """
    NNOp to triton model.
    """

    def __init__(self, model_root: str, op: 'NNOperator', model_name: str, node_conf: 'NodeConfig', server_conf: dict):
        self._model_root = model_root
        self._op = op
        self._name = model_name
        self._op_conf = node_conf.server
        self._server_conf = server_conf
        self._triton_files = TritonFiles(model_root, self._name)
        self._model_format_priority = self._server_conf.get(constant.FORMAT_PRIORITY, [])
        self._backend = None

    def _create_model_dir(self) -> bool:
        self._triton_files.root.mkdir(parents=True, exist_ok=False)
        self._triton_files.model_path.mkdir(parents=True, exist_ok=False)
        return True

    def _prepare_config(self) -> bool:
        """
        All model open with auto-generated model configuration, such as:
        server: {
             'device_ids': [0, 1],
             'max_batch_size': 128,
             'batch_latency_micros': 100000,
             'num_instances_per_device': 3,
             'triton': {
                 'preferred_batch_size': [8, 16],
             }
        }
        """
        if self._op_conf is None:
            config_lines = create_modelconfig(self._name, None, None, None, self._backend, True, None, None, None, None)
        else:
            config_lines = create_modelconfig(
                model_name=self._name,
                max_batch_size=self._op_conf.max_batch_size,
                inputs=None,
                outputs=None,
                backend=self._backend,
                enable_dynamic_batching=True,
                preferred_batch_size=self._op_conf.triton.preferred_batch_size,
                max_queue_delay_microseconds=self._op_conf.batch_latency_micros,
                instance_count=self._op_conf.num_instances_per_device,
                device_ids=self._op_conf.device_ids
            )
        with open(self._triton_files.config_file, 'wt', encoding='utf-8') as f:
            f.writelines(config_lines)
        return True

    def _save_onnx_model(self):
        self._backend = 'onnxruntime'
        try:
            if self._prepare_config() and self._op.save_model('onnx', self._triton_files.onnx_model_file):
                return 1
        except Exception as e:  # pylint: disable=broad-except
            st_err = '{}, {}'.format(str(e), traceback.format_exc())
            engine_log.error('Save the operator: %s with onnx model failed, error: %s', self._name, st_err)
            return -1

    def _prepare_model(self):
        for fmt in self._model_format_priority:
            if fmt in self._op.supported_formats:
                if fmt == 'onnx':
                    return self._save_onnx_model()
            else:
                engine_log.warning('Unknown optimize %s', fmt)
                continue
        return 0

    def get_model_in_out(self):
        model = onnx.load(str(self._triton_files.onnx_model_file))
        graph_outputs = model.graph.output
        graph_inputs = model.graph.input
        inputs = [graph_input.name for graph_input in graph_inputs]
        outputs = [graph_output.name for graph_output in graph_outputs]
        return inputs, outputs

    def to_triton(self) -> int:
        """
        save model in triton and return the status.
        0 means do nothing, -1 means failed, 1 means successfully created the model repo.
        """
        if not set(self._model_format_priority) & set(self._op.supported_formats):
            return 0
        if not self._create_model_dir():
            return -1
        return self._prepare_model()
