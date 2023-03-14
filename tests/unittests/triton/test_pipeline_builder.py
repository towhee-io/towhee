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
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from towhee.runtime import pipe, ops, AutoConfig
from towhee.runtime.dag_repr import DAGRepr
from towhee.serve.triton.constant import PIPELINE_NAME
from towhee.serve.triton.pipeline_builder import Builder
from towhee.utils.thirdparty.dill_util import dill as pickle


class TestPipelineBuilder(unittest.TestCase):
    """
    test pipeline to triton model
    """
    def test_onnx(self):
        node_config = AutoConfig.TritonGPUConfig(device_ids=[0, 1],
                                                 num_instances_per_device=3,
                                                 max_batch_size=128,
                                                 batch_latency_micros=100000,
                                                 preferred_batch_size=[8, 16])
        p = (
            pipe.input('path')
            .map('path', 'img', ops.image_decode.cv2())
            .map('img', 'embedding', ops.towhee.test_resnet18(), config=node_config)
            .output('embedding')
        )
        server_config = {
            'format_priority': ['onnx']
        }

        dag_repr = copy.deepcopy(p.dag_repr)
        model_name = 'towhee.test-resnet18-1'
        with TemporaryDirectory(dir='.') as root:
            self.assertTrue(Builder(dag_repr, root, server_config).build())

            pipe_dag_file = root + '/' + PIPELINE_NAME + '/1/pipe.pickle'
            pipe_model_file = root + '/' + PIPELINE_NAME + '/1/model.py'
            self.assertTrue(Path(pipe_dag_file).exists())
            self.assertTrue(Path(pipe_model_file).exists())
            self.assertTrue((Path(root) / model_name / '1/model.onnx').exists())
            self.assertTrue((Path(root) / model_name / 'config.pbtxt').exists())

            with open(str(pipe_dag_file), 'rb') as f_dag:
                dag_repr = pickle.load(f_dag)
                for _, node in dag_repr.nodes.items():
                    if node.config.acc_conf is not None:
                        self.assertEqual(node.config.acc_conf.triton.model_name, model_name)
                        self.assertEqual(node.config.server_conf.device_ids, [0, 1])
                        self.assertEqual(node.config.server_conf.num_instances_per_device, 3)
                        self.assertEqual(node.config.server_conf.max_batch_size, 128)
                        self.assertEqual(node.config.server_conf.batch_latency_micros, 100000)
                        self.assertEqual(node.config.server_conf.triton.preferred_batch_size, [8, 16])

    def test_normal(self):
        p = (
            pipe.input('nums', 'arr')
            .flat_map('nums', 'num', lambda x: x)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )
        config = {
            'format_priority': ['onnx']
        }

        dag_repr = copy.deepcopy(p.dag_repr)
        with TemporaryDirectory(dir='.') as root:
            self.assertTrue(Builder(dag_repr, root, config).build())

            pipe_dag_file = root + '/' + PIPELINE_NAME + '/1/pipe.pickle'
            pipe_model_file = root + '/' + PIPELINE_NAME + '/1/model.py'
            self.assertTrue(Path(pipe_dag_file).exists())
            self.assertTrue(Path(pipe_model_file).exists())

    def test_failed(self):
        dag_dict = {
            '_input': {
                'inputs': None,
                'outputs': ('a',),
                'iter_info': {
                    'type': 'map',
                    'param': None
                },
                'op_info': {
                    'operator': 'NOPNodeOperator',
                    'type': 'built_in',
                    'init_args': None,
                    'init_kws': None,
                    'tag': None
                },
                'next_nodes': ['op1']
            },
            'op1': {
                'inputs': ('a',),
                'outputs': ('b',),
                'iter_info': {
                    'type': 'map',
                    'param': None
                },
                'op_info': {
                    'operator': 'image-decode/cv2',
                    'type': 'hub',
                    'init_args': ('x',),
                    'init_kws': {},
                    'tag': 'main',
                },
                'config': None,
                'next_nodes': ['_output']
            },
            '_output': {
                'inputs': ('b',),
                'outputs': ('b',),
                'iter_info': {
                    'type': 'map',
                    'param': None
                },
                'op_info': {
                    'operator': 'NOPNodeOperator',
                    'type': 'built_in',
                    'init_args': None,
                    'init_kws': None,
                    'tag': None
                },
                'next_nodes': None
            },
        }
        config = {
            'format_priority': ['onnx']
        }
        dag_repr = DAGRepr.from_dict(dag_dict)

        self.assertFalse(Builder(dag_repr, '.', config).build())
