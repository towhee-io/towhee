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


from towhee.runtime.runtime_conf import RuntimeConf, set_runtime_config, get_sys_config, get_accelerator
from towhee.runtime.node_config import NodeConfig


class TestRuntimeConf(unittest.TestCase):
    """
    Test runtime conf
    """

    def test_create(self):
        conf = {
            'name': 'test',
            'device': 1,
            'acc_info': {
                'type': 'triton',
                'params': {
                    'model_name': 'resnet50'
                }
            }
        }

        r_conf = RuntimeConf.from_node_config(NodeConfig.from_dict(conf))
        self.assertEqual(r_conf.sys_config.device_id, 1)
        self.assertTrue(r_conf.accelerator.is_triton())
        self.assertTrue(r_conf.accelerator.triton.model_name, 'resnet50')
        
    def test_create_exception(self):
        conf = {
            'name': 'test',
            'device': 1,
            'acc_info': {
                'type': 'unkown',
                'params': {
                    'model_name': 'resnet50'
                }
            }
        }        

        with self.assertRaises(ValueError):
            RuntimeConf.from_node_config(NodeConfig.from_dict(conf))

    def test_data_trans(self):

        def inner(device_id, model_name):
            sys_conf = get_sys_config()
            acc = get_accelerator()
            assert sys_conf.device_id == device_id
            assert acc.triton.model_name == model_name

        for i in range(10):
            conf = {
                'name': 'test',
                'device': i,
                'acc_info': {
                    'type': 'triton',
                    'params': {
                        'model_name': str(i + 10)
                    }
                }
            }

            with set_runtime_config(NodeConfig.from_dict(conf)):
                inner(i, str(i + 10))
