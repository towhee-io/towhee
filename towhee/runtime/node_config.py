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

from typing import Dict, Any, Optional

from towhee.runtime.check_utils import check_config


class NodeConfig:
    """
    The config of nodes.
    """
    def __init__(self, *, name: str,
                 device: int,
                 acc_info: Optional[Dict]):
        self._name = name
        self._device = device
        self._acc_conf = AcceleratorConf.from_dict(acc_info) if acc_info is not None else None

    @property
    def name(self):
        return self._name

    @property
    def device(self):
        return self._device

    @property
    def acc_conf(self):
        return self._acc_conf

    @staticmethod
    def from_dict(conf: Dict[str, Any]):
        essentials = {'name'}
        check_config(conf, essentials)
        return NodeConfig(
            name=conf['name'],
            device=conf.get('device', -1),
            acc_info=conf.get('acc_info')
        )


class AcceleratorConf:
    """
    AcceleratorConf
    """
    def __init__(self, acc_type: str, conf: Dict):
        self._type = acc_type
        if self._type == 'triton':
            self._conf = TritonConf.from_dict(conf)
        else:
            raise ValueError(f'Unkown accelerator: {acc_type}')

    def is_triton(self):
        return self._type == 'triton'

    @property
    def triton(self):
        return self._conf

    @staticmethod
    def from_dict(acc_conf: Dict[str, Any]):
        return AcceleratorConf(acc_conf['type'], acc_conf['params'])


class TritonConf:
    """
    Triton client config.
    """
    def __init__(self, model_name: str):
        self._model_name = model_name

    @property
    def model_name(self):
        return self._model_name

    @staticmethod
    def from_dict(conf):
        if 'model_name' not in conf:
            raise ValueError('Triton accelerator lost model_name config')
        return TritonConf(conf['model_name'])
