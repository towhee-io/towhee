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

from towhee.utils.log import engine_log
from towhee.utils.onnx_utils import onnx
from towhee.serve.triton import constant

from .triton_files import TritonFiles


class ModelToTriton:
    """
    NNOp to triton model.
    """

    def __init__(self, model_root: str, op: 'NNOperator', node_conf: 'NodeConf', server_conf: dict):
        self._model_root = model_root
        self._op = op
        self._name = node_conf.name
        self._op_conf = node_conf.server_conf
        self._server_conf = server_conf
        self._triton_files = TritonFiles(model_root, self._name)
        self._model_format_priority = self._server_conf.get(constant.FORMAT_PRIORITY, [])
        self._backend = 'python'

    def _create_model_dir(self) -> bool:
        self._triton_files.root.mkdir(parents=True, exist_ok=True)
        self._triton_files.model_path.mkdir(parents=True, exist_ok=True)
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
            config_str = 'name: "{}"\n'.format(self._name)
            config_str += 'backend: "{}"\n'.format(self._backend)
        else:
            enable_dynamic_batching = self._op_conf.triton.preferred_batch_size is not None or self._op_conf.batch_latency_micros is not None

            config_str = self._create_model_config(
                max_batch_size=self._op_conf.max_batch_size,
                instance_count=self._op_conf.num_instances_per_device,
                device_ids=self._op_conf.device_ids,
                enable_dynamic_batching=enable_dynamic_batching,
                preferred_batch_size=self._op_conf.triton.preferred_batch_size,
                max_queue_delay_microseconds=self._op_conf.batch_latency_micros
            )
        with open(self._triton_files.config_file, 'wt', encoding='utf-8') as f:
            f.write(config_str)
            return True

    def _save_onnx_model(self):
        succ = self._op.save_model('onnx', self._triton_files.onnx_model_file)
        self._backend = 'onnxruntime'
        return succ

    def _prepare_model(self):
        succ = False
        if len(self._model_format_priority) == 0:
            self._model_format_priority = self._op.supported_formats
        for fmt in self._model_format_priority:
            if fmt in self._op.supported_formats:
                if fmt == 'onnx':
                    return self._save_onnx_model()
                else:
                    engine_log.error('Unknown optimize %s', fmt)
                    continue
        return succ

    def get_model_in_out(self):
        model = onnx.load(str(self._triton_files.onnx_model_file))
        graph_outputs = model.graph.output
        graph_inputs = model.graph.input
        inputs = [graph_input.name for graph_input in graph_inputs]
        outputs = [graph_output.name for graph_output in graph_outputs]
        return inputs, outputs

    def to_triton(self) -> bool:
        if self._create_model_dir() and self._prepare_model() and self._prepare_config():
            return True
        return False

    def _create_model_config(self, max_batch_size,
                             instance_count, device_ids,
                             enable_dynamic_batching, preferred_batch_size, max_queue_delay_microseconds):
        config = 'name: "{}"\n'.format(self._name)
        config += 'backend: "{}"\n'.format(self._backend)
        if max_batch_size is not None:
            config += 'max_batch_size: {}\n'.format(max_batch_size)

        if enable_dynamic_batching:
            config += '''
dynamic_batching {
'''
            if preferred_batch_size is not None:
                config += '''
    preferred_batch_size: {}
'''.format(preferred_batch_size)
            if max_queue_delay_microseconds is not None:
                config += '''
    max_queue_delay_microseconds: {}
'''.format(max_queue_delay_microseconds)
            config += '''
}\n'''

        if device_ids is not None:
            instance_count = 1 if instance_count is None else instance_count
            config += '''
instance_group [
    {{
        kind: KIND_GPU
        count: {}
        gpus: {}
    }}
]\n'''.format(instance_count, device_ids)
        elif instance_count is not None:
            config += '''
instance_group [
    {{
        kind: KIND_CPU
        count: {}
    }}
]\n'''.format(instance_count)

        return config
