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
import json


class OpConfig:
    '''
    Operator meta info.
    '''
    def __init__(self, hub: str, name: str, init_args: Dict):
        self._hub = hub
        self._op_name = name
        self._init_args = init_args

    @property
    def op_name(self):
        return self._op_name

    @property
    def op_hub(self):
        return self._hub

    @property
    def init_args(self):
        return self._init_args

    @staticmethod
    def load_from_dict(hub: str, name: str, init_args: Dict) -> 'OpConfig':
        return OpConfig(hub, name, init_args)

    @staticmethod
    def load_from_file(config_file: str) -> 'OpConfig':
        with open(config_file, 'rt', encoding='utf-8') as f:
            data = f.read()
            config_data = json.loads(data)
            return OpConfig(config_data['hub'], config_data['name'], config_data['init_args'])
