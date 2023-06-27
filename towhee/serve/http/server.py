# Copyright 2023 Zilliz. All rights reserved.
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

from typing import Callable
import inspect
from functools import partial

from towhee.utils.thirdparty.fastapi_utils import fastapi
from towhee.utils.thirdparty.uvicorn_util import uvicorn

from towhee.serve.io import JSON


class HTTPServer:
    """
    An HTTP server implemented based on FastAPI
    """

    def __init__(self, api_service: 'APIService'):
        self._app = fastapi.FastAPI()

        @self._app.get('/')
        def index():
            return api_service.desc

        def func_wrapper(func: Callable,
                         input_model: 'IOBase',
                         output_model: 'IOBase',
                         request: fastapi.Request):
            if input_model is None:
                input_model = JSON()

            if output_model is None:
                output_model = JSON()

            values = input_model.from_http(request)
            signature = inspect.signature(func)
            if len(signature.parameters.keys()) > 1:
                if isinstance(values, dict):
                    ret = output_model.to_http(func(**values))
                else:
                    ret = output_model.to_http(func(*values))
            elif len(signature.parameters.keys()) == 1:
                ret = output_model.to_http(func(values))
            else:
                ret = output_model.to_http(func())
            return ret

        for router in api_service.routers:
            wrapper = partial(func_wrapper, router.func, router.input_model, router.output_model)
            wrapper.__name__ = wrapper.func.__name__
            wrapper.__doc__ = wrapper.func.__doc__
            self._app.add_api_route(
                router.path,
                wrapper,
                methods=['POST']
            )

    @property
    def app(self):
        return self._app

    def run(self, host: str, port: int):
        uvicorn.run(self._app, host=host, port=port)
