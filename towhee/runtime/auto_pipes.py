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

from functools import wraps
from typing import Optional
import threading
import contextvars
import contextlib


import towhee.runtime.pipeline_loader as pipe_loader
from towhee.utils.log import engine_log


_AUTO_PIPES_VAR: contextvars.ContextVar = contextvars.ContextVar('auto_pipes_var')


@contextlib.contextmanager
def set_pipe_name(name: str):
    token = _AUTO_PIPES_VAR.set(name)
    yield
    _AUTO_PIPES_VAR.reset(token)


def get_pipe_name():
    try:
        return _AUTO_PIPES_VAR.get()
    except:  # pylint: disable=bare-except
        return None


class AutoPipes:
    """
    Load Predefined pipeines.
    """
    _PIPES_DEF = {}
    _lock = threading.Lock()

    def __init__(self):
        raise EnvironmentError(
            'AutoPipes is designed to be instantiated, please using the `AutoPipes.load_pipeline(pipeline_name)` etc.'
        )

    @staticmethod
    def register():
        def decorate(pipe_def):
            @wraps(pipe_def)
            def wrapper(*args, **kwargs):
                return pipe_def(*args, **kwargs)
            name = get_pipe_name()
            if name is not None:
                AutoPipes._PIPES_DEF[name] = wrapper
            return wrapper
        return decorate

    @staticmethod
    def pipeline(name, *args, **kwargs) -> Optional['RuntimePipeline']:
        with AutoPipes._lock:
            if name in AutoPipes._PIPES_DEF:
                return AutoPipes._PIPES_DEF[name](*args, **kwargs)

            with set_pipe_name(name):
                pipe_loader.PipelineLoader.load_pipeline(name)
            if name in AutoPipes._PIPES_DEF:
                return AutoPipes._PIPES_DEF[name](*args, **kwargs)

            engine_log.error('Can not found the pipline %s', name)
            return None
