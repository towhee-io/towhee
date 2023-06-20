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

from towhee.utils.thirdparty.grpc_utils import grpc

from . import service_pb2
from . import service_pb2_grpc

from towhee.serve.io import JSON, TEXT


class Response(BaseModel):
    code: int
    msg: str
    content: T.Any


class Client:
    """
    GRPCServer sync client.
    """

    def __init__(self, host: str, port: str):
        self._channel = grpc.insecure_channel(host + ':' + str(port))
        self._stub = service_pb2_grpc.PipelineServicesStub(self._channel)

    def _gen_input(self, name: str, params: T.Any, model: 'IOBase') -> 'Request':
        return service_pb2.Request(path=name, content=model.to_proto(params))

    def _parse_output(self, content: 'service_pb2.Content'):
        field_name = content.WhichOneof('content')
        if field_name == 'json':
            return JSON().from_proto(content)
        elif field_name == 'text':
            return TEXT().from_proto(content)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        self.close()

    def __call__(self, name: str, params: T.Any, model: T.Optional['IOBase'] = None):
        if model is None:
            model = JSON()

        msg = self._gen_input(name, params, model)
        response = self._stub.Predict(msg)
        if response.code == 0:
            return Response(code=response.code,
                            msg=response.msg,
                            content=self._parse_output(response.content))
        return Response(code=response.code, msg=response.msg, content=None)

    def close(self):
        self._channel.close()
