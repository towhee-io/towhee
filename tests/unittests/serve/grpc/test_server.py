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

import unittest
import typing as T
from pydantic import BaseModel
import numpy as np

from towhee import api_service, pipe
from towhee.serve.grpc.server import GRPCServer
from towhee.serve.grpc.client import Client
from towhee.serve.io import JSON, NDARRAY, BYTES
from towhee.utils.serializer import from_json


class TestGRPC(unittest.TestCase):
    """
    GRPCServer test
    """
    def tearDown(self):
        self.server.stop()

    def test_default(self):
        self.server = GRPCServer(
            api_service.build_service(
                [
                    (lambda x: x, '/echo'),
                    (lambda x: x + 1, '/add_one')
                ]
            )
        )
        self.server.start('localhost', 50001)

        client = Client('localhost', 50001)
        response = client('/echo', 1)
        self.assertEqual(from_json(response.content), 1)

        response = client('/add_one', 1)
        self.assertEqual(from_json(response.content), 2)

    def test_with_model(self):
        class Item(BaseModel):
            url: str
            ids: T.List[int]

        service = api_service.APIService(desc='test')

        @service.api(path='/test', input_model = JSON(Item), output_model=JSON(Item))
        def test(item: Item):
            return item

        self.server = GRPCServer(service)
        self.server.start('localhost', 50001)
        client = Client('localhost', 50001)

        data = Item(
            url='test_url',
            ids=[1, 2, 3]
        )

        response = client('/test', data.dict())
        self.assertEqual(response.content, data.dict())

    def test_muilt_params(self):
        service = api_service.APIService(desc='test')

        @service.api(path='/test')
        def test(url: str, ids: T.List[int]):
            return {'url': url, 'ids': ids}

        @service.api(path='/test_list', output_model=JSON())
        def test_list(url: str, ids: T.List[int]):
            return [url, ids]

        self.server = GRPCServer(service)
        self.server.start('localhost', 50001)

        client = Client('localhost', 50001)
        data = {'url': 'my_test_url', 'ids': [2, 3, 4]}
        response = client('/test', data)
        self.assertEqual(from_json(response.content), data)

        data = ['my_test_url', [2, 3, 4]]
        response = client('/test_list', data)
        self.assertEqual(response.content, data)

        # test not exist path
        response = client('/not_exist', '')
        self.assertEqual(response.code, -1)

    def test_pipeline(self):
        self.server = GRPCServer(
            api_service.build_service(
                (pipe.input('nums').map('nums', 'sum', sum).output('sum'), '/sum')
            )
        )

        self.server.start('localhost', 50001)
        client = Client('localhost', 50001)
        response = client('/sum', [1, 2, 3])
        content = from_json(response.content)
        self.assertEqual(content[0][0], 6)
        response = client('/sum/batch', [[1, 2, 3], [2, 3, 4], [4, 5, 6]])
        content = from_json(response.content)
        self.assertEqual(content[0][0][0], 6)
        self.assertEqual(content[1][0][0], 9)
        self.assertEqual(content[2][0][0], 15)

    def test_numpy(self):
        p = (
            pipe.input('nums')
            .flat_map('nums', 'num', lambda x: x)
            .map('nums', 'nums', lambda x: np.random.rand(1024))
            .output('nums', 'num')
        )

        self.server = GRPCServer(
            api_service.build_service(
                (p, '/numpys')
            )
        )

        self.server.start('localhost', 50001)
        client = Client('localhost', 50001)

        response = client('/numpys', [1, 2, 3])
        content = from_json(response.content)
        self.assertEqual(content[0][1], 1)
        self.assertEqual(content[1][1], 2)
        self.assertEqual(content[2][1], 3)

    def test_with(self):
        self.server = GRPCServer(
            api_service.build_service(
                (pipe.input('nums').map('nums', 'sum', sum).output('sum'), '/sum')
            )
        )

        self.server.start('localhost', 50001)
        with Client('localhost', 50001) as client:
            response = client('/sum', [1, 2, 3])
            self.assertEqual(response.code, 0)
            content = from_json(response.content)
            self.assertEqual(content[0][0], 6)

    def test_ndarray_io(self):
        service = api_service.APIService(desc='test')

        @service.api(path='/test', output_model=NDARRAY())
        def test():
            return np.random.rand(1024)

        @service.api(path='/echo', input_model=NDARRAY(), output_model=NDARRAY())
        def test2(arr: 'ndarray'):
            return arr

        @service.api(path='/unkown_model', output_model=NDARRAY())
        def test3(arr: 'ndarray'):
            return arr

        self.server = GRPCServer(service)

        self.server.start('localhost', 50001)
        client = Client('localhost', 50001)
        with Client('localhost', 50001) as client:
            response = client('/test')
            self.assertTrue(isinstance(response.content, np.ndarray))

            arr = np.random.rand(1024)
            response = client('/echo', arr, NDARRAY())
            self.assertTrue(np.array_equal(arr, response.content))

            response = client('/echo', arr)
            self.assertTrue(np.array_equal(arr, response.content))

            response = client('/unkown_model', arr)
            self.assertTrue(np.array_equal(arr, response.content))

            with self.assertRaises(AssertionError):
                client('/echo', 'error_type', NDARRAY())

    def test_bytes_io(self):
        test_str = 'Hello towhee'
        b = bytes(test_str, 'utf-8')

        service = api_service.APIService(desc='test')

        @service.api(path='/echo', input_model=BYTES(), output_model=BYTES())
        def test2(b: bytes):
            assert isinstance(b, bytes)
            s = b.decode('utf-8')
            assert s == test_str
            return bytes('welcome', 'utf-8')

        self.server = GRPCServer(service)
        self.server.start('localhost', 50001)

        with Client('localhost', 50001) as client:
            response = client('/echo', b, BYTES())
            self.assertEqual(b'welcome', response.content)

            response = client('/echo', b)
            self.assertEqual(b'welcome', response.content)

            with self.assertRaises(AssertionError):
                client('/echo', 'error_type', NDARRAY())
