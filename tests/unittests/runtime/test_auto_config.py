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

from towhee.dc2 import AutoConfig


class TestAutoConfig(unittest.TestCase):
    """
    Test AutoConfig
    """

    def test_local(self):
        conf1 = AutoConfig.LocalCPUConfig()
        self.assertEqual(conf1, [{'device': -1}])

        conf2 = AutoConfig.LocalGPUConfig()
        self.assertEqual(conf2, [{'device': 0}])

        conf3 = AutoConfig.LocalGPUConfig(device=1)
        self.assertEqual(conf3, [{'device': 1}])

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
        self.assertEqual(conf1, [conf1_res])

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
        self.assertEqual(conf2, [conf2_res])

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
        self.assertEqual(conf3, [conf3_res])

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
        self.assertEqual(conf4, [conf4_res])

    def test_multi(self):
        conf1 = AutoConfig.LocalGPUConfig() + AutoConfig.TritonGPUConfig()
        local_conf = {'device': 0}
        triton_conf = {
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
        self.assertEqual(conf1, [local_conf, triton_conf])
