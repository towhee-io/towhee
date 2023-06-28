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

import typing as T
from pydantic import BaseModel
import numpy as np

from towhee.utils.thirdparty.grpc_utils import grpc

from . import service_pb2
from . import service_pb2_grpc

from towhee.serve.io import JSON, TEXT, NDARRAY, BYTES


class Response(BaseModel):
    code: int
    msg: str
    content: T.Any


def _parse_output(content: 'service_pb2.Content'):
    field_name = content.WhichOneof('content')
    if field_name == 'content_bytes':
        return BYTES().from_proto(content)
    elif field_name == 'text':
        return TEXT().from_proto(content)
    elif field_name == 'tensor':
        return NDARRAY().from_proto(content)
    else:
        return JSON().from_proto(content)


def _gen_input(name: str, params: T.Any, model: 'IOBase') -> 'Request':
    if model is None:
        if isinstance(params, bytes):
            model = BYTES()
        elif isinstance(params, np.ndarray):
            model = NDARRAY()
        elif isinstance(params, str):
            model = TEXT()
        else:
            model = JSON()

    if params is None:
        return service_pb2.Request(path=name)
    return service_pb2.Request(path=name, content=model.to_proto(params))


class Client:
    """
    GRPCServer sync client.
    """

    def __init__(self, host: str, port: str):
        self._channel = grpc.insecure_channel(host + ':' + str(port))
        self._stub = service_pb2_grpc.PipelineServicesStub(self._channel)

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        self.close()

    def __call__(self, name: str, params: T.Any = None, model: T.Optional['IOBase'] = None):
        msg = _gen_input(name, params, model)
        response = self._stub.Predict(msg)
        if response.code == 0:
            return Response(code=response.code,
                            msg=response.msg,
                            content=_parse_output(response.content))
        return Response(code=response.code, msg=response.msg, content=None)

    def close(self):
        self._channel.close()


class AsyncClient:
    """
    GRPCServer async client.
    """

    def __init__(self, host: str, port: str):
        self._channel = grpc.aio.insecure_channel(host + ':' + str(port))
        self._stub = service_pb2_grpc.PipelineServicesStub(self._channel)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_details):
        await self.close()

    async def __call__(self, name: str, params: T.Any = None, model: T.Optional['IOBase'] = None):
        msg = _gen_input(name, params, model)
        response = await self._stub.Predict(msg)
        if response.code == 0:
            return Response(code=response.code,
                            msg=response.msg,
                            content=_parse_output(response.content))
        return Response(code=response.code, msg=response.msg, content=None)

    async def close(self):
        await self._channel.close()
