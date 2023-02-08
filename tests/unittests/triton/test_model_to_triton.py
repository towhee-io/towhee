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

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torchvision
from towhee.runtime.node_config import NodeConfig
from towhee.serve.triton.model_to_triton import ModelToTriton


class Model:
    """
    Model class
    """
    def __init__(self):
        self.model = torchvision.models.resnet18(pretrained=True)

    def __call__(self, image):
        output_tensor = self.model(image)
        return output_tensor.detach().numpy()

    def save_model(self, model_type, output_file):
        if model_type != 'onnx':
            return False
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(self.model, dummy_input, output_file, input_names=['input0'], output_names=['output0'])
        return True

    @property
    def supported_formats(self):
        return ['onnx']


op = Model()


# pylint: disable=protected-access
class TestModelToTriton(unittest.TestCase):
    """
    Test ModelToTriton
    """
    def test_to_triton(self):
        name = 'image-text-embedding/resnet-18'
        server_config = {'format_priority': ['onnx', 'tensorrt']}
        node_config = NodeConfig.from_dict({'name': name})
        model_name = name.replace('/', '.')
        with TemporaryDirectory(dir='./') as root:
            m = ModelToTriton(root, op, model_name, node_config, server_config)
            self.assertEqual(m.to_triton(), 1)

            inputs, outputs = m.get_model_in_out()
            self.assertTrue(inputs, ['input0'])
            self.assertTrue(outputs, ['output0'])

            model_path = Path(root) / model_name
            path1 = model_path / 'config.pbtxt'
            with open(path1, encoding='utf-8') as f:
                file_config = list(f.readlines())
            pbtxt_config = ['name: "image-text-embedding.resnet-18"\n',
                            'backend: "onnxruntime"\n', '\n',
                            'dynamic_batching {\n', '\n', '}\n', '\n',
                            'instance_group [\n',
                            '    {\n',
                            '        kind: KIND_CPU\n',
                            '        count: 1\n',
                            '    }\n',
                            ']\n']
            self.assertEqual(file_config, pbtxt_config)
            path2 = model_path / '1' / 'model.onnx'
            self.assertTrue(path2.exists())

    def test_prepare(self):
        name = 'resnet18_conf1'
        server_config = {'format_priority': ['onnx', 'tensorrt']}
        node_config = NodeConfig.from_dict({
            'name': name,
            'server': {
                'device_ids': [0, 1],
                'max_batch_size': 128,
                'batch_latency_micros': 100000,
                'no_key': 0,
                'triton': {
                    'preferred_batch_size': [8, 16],
                }
            }
        })
        with TemporaryDirectory(dir='./') as root:
            m = ModelToTriton(root, op, name, node_config, server_config)
            self.assertEqual(m.to_triton(), 1)

            model_path = Path(root) / name
            path1 = model_path / 'config.pbtxt'
            with open(path1, encoding='utf-8') as f:
                file_config = list(f.readlines())
            pbtxt_config = ['name: "resnet18_conf1"\n',
                            'backend: "onnxruntime"\n',
                            'max_batch_size: 128\n', '\n',
                            'dynamic_batching {\n', '\n',
                            '    preferred_batch_size: [8, 16]\n', '\n',
                            '    max_queue_delay_microseconds: 100000\n', '\n',
                            '}\n', '\n',
                            'instance_group [\n',
                            '    {\n',
                            '        kind: KIND_GPU\n',
                            '        count: 1\n',
                            '        gpus: [0, 1]\n',
                            '    }\n',
                            ']\n']
            self.assertEqual(file_config, pbtxt_config)
            path2 = model_path / '1' / 'model.onnx'
            self.assertTrue(path2.exists())

    def test_status(self):
        name = 'resnet18_conf2'
        server_config = {}
        node_config = NodeConfig.from_dict({
            'name': name,
            'server': {'num_instances_per_device': 3}
        })
        with TemporaryDirectory(dir='./') as root:
            path2 = Path(root)
            m = ModelToTriton(root, op, name, node_config, server_config)
            self.assertEqual(m.to_triton(), 0)
            self.assertFalse((path2 / name).exists())

        name = 'resnet18_conf3'
        server_config = {'format_priority': ['tensorrt']}
        node_config = NodeConfig.from_dict({
            'name': name,
            'server': {'num_instances_per_device': 3}
        })
        with TemporaryDirectory(dir='./') as root:
            path3 = Path(root)
            m = ModelToTriton(root, op, name, node_config, server_config)
            self.assertEqual(m.to_triton(), 0)
            self.assertFalse((path3 / name).exists())

        server_config = {'format_priority': ['onnx']}
        op_tmp = Model()
        op_tmp.save_model = lambda x: False
        with TemporaryDirectory(dir='./') as root:
            path4 = Path(root)
            m = ModelToTriton(root, op_tmp, name, node_config, server_config)
            self.assertEqual(m.to_triton(), -1)
            self.assertTrue((path4 / name).exists())
