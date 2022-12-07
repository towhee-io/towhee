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

#pylint: disable=import-outside-toplevel
import contextvars
import contextlib
import functools


class RuntimeConf:
    """
    Runtime config, passed by contextvars
    """
    def __init__(self, sys_conf=None, acc=None):
        self._sys_config = sys_conf
        self._accelerator = acc

    @property
    def accelerator(self):
        return self._accelerator

    @property
    def sys_config(self):
        return self._sys_config

    @staticmethod
    def from_node_config(node_conf: 'NodeConfig'):
        sys_conf = SysConf(node_conf.device)
        return RuntimeConf(sys_conf, node_conf.acc_conf)


class SysConf:
    """
    sys conf
    """

    def __init__(self, device_id: int=-1):
        self._device_id = device_id

    @property
    def device_id(self):
        return self._device_id


_RUNTIME_CONF_VAR: contextvars.ContextVar = contextvars.ContextVar('runtime_conf')


@contextlib.contextmanager
def set_runtime_config(node_conf: 'NodeConfig'):
    runtime_conf = RuntimeConf.from_node_config(node_conf)
    token = _RUNTIME_CONF_VAR.set(runtime_conf)
    yield
    _RUNTIME_CONF_VAR.reset(token)


def get_sys_config():
    try:
        runtime_conf = _RUNTIME_CONF_VAR.get()
        return runtime_conf.sys_config
    except:  # pylint: disable=bare-except
        return None


def get_accelerator():
    try:
        runtime_conf = _RUNTIME_CONF_VAR.get()
        return runtime_conf.accelerator
    except:  # pylint: disable=bare-except
        return None


def accelerate(model):
    @functools.wraps(model)
    def _decorated(*args, **kwargs):
        runtime_conf = _RUNTIME_CONF_VAR.get()
        if runtime_conf.accelerator is None:
            return model(*args, **kwargs)
        elif runtime_conf.accelerator.is_triton():
            from towhee.serve.triton_client import TritonClient
            triton_conf = runtime_conf.accelerator.triton
            return TritonClient(triton_conf.model_name, triton_conf.inputs, triton_conf.outputs)
        elif runtime_conf.accelerator.is_mock():
            return MockModel()
        else:
            return None
    return _decorated


class MockModel:
    def __call__(self, data):
        return data

