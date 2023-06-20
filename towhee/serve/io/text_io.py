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
from typing import Any

from .base import IOBase, IOType
from towhee.utils.serializer import to_json


class TEXT(IOBase, io_type=IOType.TEXT):
    """
    TEXT IO
    """
    _mime_type = 'application/json'

    def from_http(self, request: 'fastapi.Request'):
        return asyncio.run(request.body())

    def to_http(self, obj: Any) -> 'Response':
        from fastapi import Response  # pylint: disable=import-outside-toplevel

        if not isinstance(obj, str):
            obj = to_json(obj)
        return Response(content=obj, media_type=self._mime_type)

    def from_proto(self, content: 'service_pb2.Content') -> str:
        return content.text

    def to_proto(self, obj: Any) -> 'service_pb2.Content':
        from towhee.serve.grpc import service_pb2  # pylint: disable=import-outside-toplevel

        if not isinstance(obj, str):
            obj = to_json(obj)
        return service_pb2.Content(text=obj)
