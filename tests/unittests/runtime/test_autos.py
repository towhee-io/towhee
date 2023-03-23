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

import pathlib
import unittest

from towhee import AutoConfig, AutoPipes
from towhee.runtime.node_config import TowheeConfig


class TestAutoConfig(unittest.TestCase):
    """
    Test AutoConfig
    """

    def test_local(self):
        conf1 = AutoConfig.LocalCPUConfig()
        self.assertTrue(isinstance(conf1, TowheeConfig))
        self.assertEqual(conf1.config, {'device': -1})

        conf2 = AutoConfig.LocalGPUConfig()
        self.assertTrue(isinstance(conf2, TowheeConfig))
        self.assertEqual(conf2.config, {'device': 0})

        conf3 = AutoConfig.LocalGPUConfig(device=1)
        self.assertTrue(isinstance(conf3, TowheeConfig))
        self.assertEqual(conf3.config, {'device': 1})

    def test_Triton(self):
        conf1 = AutoConfig.TritonCPUConfig()
        conf1_res = {
            'server': {
                'device_ids': None,
                'num_instances_per_device': 1,
                'max_batch_size': None,
                'batch_latency_micros': None,
                'triton': {
                    'preferred_batch_size': None
                }
            }
        }
        self.assertTrue(isinstance(conf1, TowheeConfig))
        self.assertEqual(conf1.config, conf1_res)

        conf2 = AutoConfig.TritonCPUConfig(num_instances_per_device=2,
                                           max_batch_size=128,
                                           batch_latency_micros=100000,
                                           preferred_batch_size=[8, 16])
        conf2_res = {
            'server': {
                'device_ids': None,
                'num_instances_per_device': 2,
                'max_batch_size': 128,
                'batch_latency_micros': 100000,
                'triton': {
                    'preferred_batch_size': [8, 16]
                }
            }
        }
        self.assertTrue(isinstance(conf2, TowheeConfig))
        self.assertEqual(conf2.config, conf2_res)

        conf3 = AutoConfig.TritonGPUConfig()
        conf3_res = {
            'server': {
                'device_ids': [0],
                'num_instances_per_device': 1,
                'max_batch_size': None,
                'batch_latency_micros': None,
                'triton': {
                    'preferred_batch_size': None
                }
            }
        }
        self.assertTrue(isinstance(conf3, TowheeConfig))
        self.assertEqual(conf3.config, conf3_res)

        conf4 = AutoConfig.TritonGPUConfig(device_ids=[0, 1],
                                           num_instances_per_device=2,
                                           max_batch_size=128,
                                           batch_latency_micros=100000,
                                           preferred_batch_size=[8, 16])
        conf4_res = {
            'server': {
                'device_ids': [0, 1],
                'num_instances_per_device': 2,
                'max_batch_size': 128,
                'batch_latency_micros': 100000,
                'triton': {
                    'preferred_batch_size': [8, 16]
                }
            }
        }
        self.assertTrue(isinstance(conf4, TowheeConfig))
        self.assertEqual(conf4.config, conf4_res)

    def test_multi(self):
        conf1 = AutoConfig.LocalGPUConfig() + AutoConfig.TritonGPUConfig()
        dict_conf = {
            'device': 0,
            'server': {
                'device_ids': [0],
                'num_instances_per_device': 1,
                'max_batch_size': None,
                'batch_latency_micros': None,
                'triton': {
                    'preferred_batch_size': None
                }
            }
        }
        self.assertTrue(isinstance(conf1, TowheeConfig))
        self.assertEqual(conf1.config, dict_conf)


class TestAutoPipes(unittest.TestCase):
    """
    Test load pipeline
    """
    def test_local(self):
        file_path = str(pathlib.Path(__file__).parent / 'builtin_pipeline.py')
        config = AutoConfig.load_config(file_path)
        p = AutoPipes.pipeline(file_path, config)
        ret = p(10).get()[0]
        self.assertEqual(ret, 11)

    def test_builtin(self):
        conf = AutoConfig.load_config('_builtin_pipeline')

        conf.param = 2
        p = AutoPipes.pipeline('_builtin_pipeline', conf)
        ret = p(10).get()[0]
        self.assertEqual(ret, 12)

    def test_local_repo(self):
        conf = AutoConfig.load_config('local/ci-test2')
        conf.param = 1
        p = AutoPipes.pipeline('local/ci-test2', conf)
        self.assertEqual(p(1).get(), [2])

    def test_hub_repo(self):
        conf = AutoConfig.load_config('towhee/test-revision-pipeline')
        conf.param = 10
        p = AutoPipes.pipeline('towhee/test-revision-pipeline', conf)
        self.assertEqual(p(1).get(), [11])
