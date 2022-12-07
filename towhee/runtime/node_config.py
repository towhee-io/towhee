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

from typing import Dict, Any, Optional, List

from towhee.runtime.check_utils import check_config, check_supported


class NodeConfig:
    """
    The config of nodes.
    """
    def __init__(self, *, name: str,
                 device: int,
                 acc_info: Optional[Dict],
                 server_info: Optional[Dict]):
        self._name = name
        self._device = device
        self._acc_conf = AcceleratorConf.from_dict(acc_info) if acc_info is not None else None
        self._server_conf = ServerConf.from_dict(server_info) if server_info is not None else None

    @property
    def name(self):
        return self._name

    @property
    def device(self):
        return self._device

    @property
    def acc_conf(self):
        return self._acc_conf

    @property
    def server_conf(self):
        return self._server_conf

    @staticmethod
    def from_dict(conf: Dict[str, Any]):
        essentials = {'name'}
        check_config(conf, essentials)
        return NodeConfig(
            name=conf['name'],
            device=conf.get('device', -1),
            acc_info=conf.get('acc_info'),
            server_info=conf.get('server')
        )


class AcceleratorConf:
    """
    AcceleratorConf
    """
    def __init__(self, acc_type: str, conf: Dict):
        self._type = acc_type
        if self._type == 'triton':
            self._conf = TritonClientConf.from_dict(conf)
        elif self._type == 'mock':
            pass
        else:
            raise ValueError(f'Unkown accelerator: {acc_type}')

    def is_triton(self):
        return self._type == 'triton'

    def is_mock(self):
        return self._type == 'mock'

    @property
    def triton(self):
        return self._conf

    @staticmethod
    def from_dict(acc_conf: Dict[str, Any]):
        return AcceleratorConf(acc_conf['type'], acc_conf['params'])


class TritonClientConf:
    """
    Triton client config.
    """
    def __init__(self, model_name: str, inputs: List[str], outputs: List[str]):
        self._model_name = model_name
        self._inputs = inputs
        self._outputs = outputs

    @property
    def model_name(self):
        return self._model_name

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @staticmethod
    def from_dict(conf):
        return TritonClientConf(conf['model_name'], conf['inputs'], conf['outputs'])


class ServerConf:
    """
    ServerConf
    """
    def __init__(self, device_ids,
                 max_batch_size,
                 batch_latency_micros,
                 num_instances_per_device,
                 triton: 'TritonServerConf'):
        self._device_ids = device_ids
        self._max_batch_size = max_batch_size
        self._batch_latency_micros = batch_latency_micros
        self._num_instances_per_device = num_instances_per_device
        self._triton = triton

    @property
    def device_ids(self):
        return self._device_ids

    @property
    def max_batch_size(self):
        return self._max_batch_size

    @property
    def batch_latency_micros(self):
        return self._batch_latency_micros

    @property
    def num_instances_per_device(self):
        return self._num_instances_per_device

    @property
    def triton(self):
        return self._triton

    @staticmethod
    def from_dict(server_info: Dict[str, Any]):
        check_supported(server_info, {'device_ids', 'max_batch_size', 'batch_latency_micros', 'num_instances_per_device', 'triton'})
        triton_conf = TritonServerConf.from_dict(server_info.get('triton'))
        return ServerConf(server_info.get('device_ids'), server_info.get('max_batch_size'), server_info.get('batch_latency_micros'),
                          server_info.get('num_instances_per_device'), triton_conf)


class TritonServerConf:
    """
    Triton server config.
    """
    def __init__(self, preferred_batch_size: str = None):
        self._preferred_batch_size = preferred_batch_size

    @property
    def preferred_batch_size(self):
        return self._preferred_batch_size

    @staticmethod
    def from_dict(triton_info: Dict[str, Any]):
        if triton_info is None:
            return TritonServerConf()
        check_supported(triton_info, {'preferred_batch_size'})
        if 'preferred_batch_size' not in triton_info:
            return TritonServerConf()
        return TritonServerConf(triton_info.get('preferred_batch_size'))
