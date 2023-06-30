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

import numpy as np

from .base import IOBase, IOType
from towhee.utils.serializer import to_json, from_json


type_str_to_dtype_map = {
    'string_values': np.object_,
    'float_values': np.float32,
    'double_values': np.float64,
    'bool_values': np.bool_,
    'int32_values': np.int32,
    'int64_values': np.int64,
    'uint32_values': np.uint32,
    'uint64_values': np.uint64,
}


dtype_to_type_str_map = {
    'object': 'string_values',
    'float32': 'float_values',
    'float64': 'double_values',
    'bool': 'bool_values',
    'int32': 'int32_values',
    'int64': 'int64_values',
    'uint32': 'uint32_values',
    'unit64': 'uint64_values'
}


class NDARRAY(IOBase, io_type=IOType.NDARRAY):
    """Numpy Ndarray

    Example:
        import typing as T
        from pydantic import BaseModel
        from towhee import AutoPipes, api_service
        from towhee.serve.io import TEXT, NDARRAY

        service = api_service.APIService(desc="Welcome")
        stn = AutoPipes.pipeline('sentence_embedding')

        @service.api(path='/embedding', input_model=TEXT(), output_model=NDARRAY())
        def chat(text: str):
            return stn(text).to_list()[0][0]
    """

    _mime_type = 'application/json'

    def from_http(self, request: 'fastapi.Request') -> 'ndarray':
        json_str = asyncio.run(request.body())
        return from_json(json_str)

    def to_http(self, obj: 'ndarray') -> 'Response':
        from fastapi import Response  # pylint: disable=import-outside-toplevel

        assert isinstance(obj, np.ndarray)
        content = to_json(obj)
        return Response(content=content, media_type=self._mime_type)

    def from_proto(self, content: 'service_pb2.Content') -> 'ndarray':
        type_str = content.tensor.dtype
        shape = content.tensor.shape
        data = content.tensor.data

        array = np.array(getattr(data, type_str), type_str_to_dtype_map[type_str])
        if shape:
            array.reshape(shape)
        return array

    def to_proto(self, obj: 'ndarray') -> 'service_pb2.Content':
        from towhee.serve.grpc import service_pb2  # pylint: disable=import-outside-toplevel

        assert isinstance(obj, np.ndarray)
        dtype = str(obj.dtype)
        if dtype not in dtype_to_type_str_map:
            raise RuntimeError('Unsupport dtype: %s' % dtype)

        return service_pb2.Content(tensor=service_pb2.Tensor(
            dtype=dtype_to_type_str_map[dtype],
            shape=obj.shape,
            data=self._to_tensorcontent(obj)
        ))

    def _to_tensorcontent(self, obj: 'ndarray'):
        from towhee.serve.grpc import service_pb2  # pylint: disable=import-outside-toplevel

        if obj.dtype == np.object_:
            return service_pb2.TensorContents(
                string_values=obj
            )
        else:
            return service_pb2.TensorContents(
                **{dtype_to_type_str_map[str(obj.dtype)]: obj.ravel().tolist()}
            )
