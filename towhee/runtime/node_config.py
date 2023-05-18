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

from typing import Dict, Optional, List

from pydantic import BaseModel, Extra, validator


class TritonServerConf(BaseModel):
    """
    Triton server config.
    """
    preferred_batch_size: Optional[List[int]] = None


class ServerConf(BaseModel):
    """
    ServerConf
    """

    device_ids: Optional[List[int]] = None
    max_batch_size: Optional[int] = None
    batch_latency_micros: Optional[int] = None
    num_instances_per_device: Optional[int] = None
    triton: TritonServerConf = TritonServerConf()


class TritonClientConf(BaseModel):
    """
    Triton client config.
    """
    model_name: str
    inputs: List[str]
    outputs: List[str]


class AcceleratorConf(BaseModel):
    """
    AcceleratorConf
    """

    type: str
    params: Optional[TritonClientConf]

    @validator('type')
    @classmethod
    def type_match(cls, v):
        if v not in ['triton', 'mock']:
            raise ValueError(f'Unkown accelerator: {v}')
        return v

    def is_triton(self):
        return self.type == 'triton'

    def is_mock(self):
        return self.type == 'mock'

    @property
    def triton(self):
        return self.params


class NodeConfig(BaseModel, extra=Extra.allow):
    """
    The config of nodes.
    """

    name: str
    device: int = -1
    acc_info: Optional[AcceleratorConf] = None
    server: Optional[ServerConf] = None


class TowheeConfig:
    """
    TowheeConfig mapping for AutoConfig.
    """
    def __init__(self, config: Dict):
        self._config = config

    @property
    def config(self) -> Dict:
        return self._config

    @classmethod
    def set_local_config(cls, device: int) -> 'TowheeConfig':
        config = {
            'device': device
        }
        return cls(config)

    @classmethod
    def set_triton_config(cls,
                          device_ids: list,
                          num_instances_per_device: int,
                          max_batch_size: int,
                          batch_latency_micros: int,
                          preferred_batch_size: list) -> 'TowheeConfig':
        config = {
            'server': {
                'device_ids': device_ids,
                'num_instances_per_device': num_instances_per_device,
                'max_batch_size': max_batch_size,
                'batch_latency_micros': batch_latency_micros,
                'triton': {
                    'preferred_batch_size': preferred_batch_size
                }
            }
        }
        return cls(config)

    def __add__(self, other: 'TowheeConfig') -> 'TowheeConfig':
        if isinstance(self._config, dict):
            self._config.update(other.config)
        else:
            self._config = other.config
        return self

    def __or__(self, other: 'TowheeConfig') -> 'TowheeConfig':
        return self + other
