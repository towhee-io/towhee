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

import asyncio
from typing import Optional, Union, Any

from pydantic import BaseModel


from .base import IOBase, IOType
from towhee.utils.serializer import to_json, from_json


class JSON(IOBase, io_type=IOType.JSON):
    """JSON IO

    Example:

    .. code-block:: python

        import typing as T
        from pydantic import BaseModel
        from towhee import AutoPipes, api_service
        from towhee.serve.io import JSON

        service = api_service.APIService(desc="Welcome")
        stn = AutoPipes.pipeline('sentence_embedding')

        @service.api(path='/embedding', input_model = JSON(), output_model=JSON())
        def chat(params: T.List[T.Any]):
            return stn(*params).to_list()

        class Item(BaseModel):
            url: str
            ids: T.List[int]

        @service.api(path='/echo', input_model = JSON(Item), output_model=JSON(Item))
        def echo(item: Item):
            return item

        Client Example:
            http:

            .. code-block:: python

                import requests
                requests.post('http://127.0.0.1:8000/echo', json={'url': 1, 'ids': [1, 2]}).json()
                requests.post('http://127.0.0.1:8000/embedding', json=['hello world']).json()

            grpc:

            .. code-block:: python

                from towhee.serve.grpc.client import Client
                c = Client('0.0.0.0', 8000)
                c('/echo', {'url': 1, 'ids': [1, 2]})
                c('/embedding', ['hello'])
    """

    _mime_type = 'application/json'

    def __init__(self, model: Optional[BaseModel] = None):
        self._model = model

    def from_http(self, request: 'Request'):
        json_str = asyncio.run(request.body())
        json_obj = from_json(json_str)
        if self._model:
            return self._model.parse_obj(json_obj)
        return json_obj

    def to_http(self, obj: Union[Any, BaseModel]) -> 'Response':
        from fastapi import Response  # pylint: disable=import-outside-toplevel

        if isinstance(obj, BaseModel):
            obj = obj.dict()
        content = to_json(obj)
        return Response(content=content, media_type=self._mime_type)

    def from_proto(self, content: 'service_pb2.Content') -> Any:
        from google.protobuf.json_format import MessageToDict  # pylint: disable=import-outside-toplevel

        parsed = MessageToDict(content.json, preserving_proto_field_name=True)
        if self._model:
            return self._model.parse_obj(parsed)
        return parsed

    def to_proto(self, obj: Any) -> 'service_pb2.Content':
        from google.protobuf.json_format import ParseDict  # pylint: disable=import-outside-toplevel
        from google.protobuf import struct_pb2  # pylint: disable=import-outside-toplevel

        from towhee.serve.grpc import service_pb2 # pylint: disable=import-outside-toplevel

        msg = struct_pb2.Value()

        if isinstance(obj, BaseModel):
            obj = obj.dict()

        ParseDict(obj, msg)
        return service_pb2.Content(json=msg)
