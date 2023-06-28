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

from towhee.utils.thirdparty.fastapi_utils import fastapi  # pylint: disable=unused-import
from fastapi.testclient import TestClient

# pylint: disable=ungrouped-imports
from towhee import api_service, pipe
from towhee.serve.http.server import HTTPServer
from towhee.serve.io import JSON, NDARRAY, BYTES
from towhee.utils.serializer import to_json, from_json


class TestHTTPServer(unittest.TestCase):
    """
    Test server
    """

    def test_default(self):
        server = HTTPServer(
            api_service.build_service(
                [
                    (lambda x: x, '/echo'),
                    (lambda x: x + 1, '/add_one')
                ]
            )
        )

        client = TestClient(server.app)

        response = client.get('/')
        assert response.status_code == 200

        response = client.post('/echo', json=1)
        assert response.status_code == 200
        self.assertEqual(response.json(), 1)

        response = client.post('/add_one', json=1)
        assert response.status_code == 200
        self.assertEqual(response.json(), 2)

    def test_with_model(self):
        class Item(BaseModel):
            url: str
            ids: T.List[int]

        service = api_service.APIService(desc='test')

        @service.api(path='/test', input_model = JSON(Item), output_model=JSON(Item))
        def test(item: Item):
            return item

        server = HTTPServer(service)
        client = TestClient(server.app)
        response = client.get('/')
        assert response.status_code == 200

        data = Item(
            url='test_url',
            ids=[1, 2, 3]
        )

        response = client.post('/test', json=data.dict())
        assert response.status_code == 200
        self.assertEqual(response.json(), data.dict())

    def test_muilt_params(self):
        service = api_service.APIService(desc='test')

        @service.api(path='/test')
        def test(url: str, ids: T.List[int]):
            return {'url': url, 'ids': ids}

        @service.api(path='/test_list')
        def test_list(url: str, ids: T.List[int]):
            return [url, ids]

        server = HTTPServer(service)
        client = TestClient(server.app)
        response = client.get('/')
        assert response.status_code == 200

        data = {'url': 'my_test_url', 'ids': [2, 3, 4]}
        response = client.post('/test', json=data)
        assert response.status_code == 200
        self.assertEqual(response.json(), data)

        data = ['my_test_url', [2, 3, 4]]
        response = client.post('/test_list', json=data)
        assert response.status_code == 200
        self.assertEqual(response.json(), data)


    def test_pipeline(self):
        server = HTTPServer(
            api_service.build_service(
                (pipe.input('nums').map('nums', 'sum', sum).output('sum'), '/sum')
            )
        )

        client = TestClient(server.app)

        response = client.get('/')
        assert response.status_code == 200

        response = client.post('/sum', json=[1, 2, 3])
        assert response.status_code == 200
        self.assertEqual(response.json()[0][0], 6)

        response = client.post('/sum/batch', json=[[1, 2, 3], [2, 3, 4], [4, 5, 6]])
        assert response.status_code == 200
        self.assertEqual(response.json()[0][0][0], 6)
        self.assertEqual(response.json()[1][0][0], 9)
        self.assertEqual(response.json()[2][0][0], 15)

    def test_ndarray_io(self):
        service = api_service.APIService(desc='test')

        arr = np.random.rand(1024)

        @service.api(path='/echo', output_model=NDARRAY())
        def test(arr: 'ndarray'):
            return arr

        server = HTTPServer(service)
        client = TestClient(server.app)
        response = client.post('/echo', data=to_json(arr))
        self.assertTrue(isinstance(from_json(response.content), np.ndarray))

    def test_bytes_io(self):
        service = api_service.APIService(desc='test')

        @service.api(path='/echo', input_model=BYTES(), output_model=BYTES())
        def test(b: bytes):
            assert isinstance(b, bytes)
            return b'welcome'

        server = HTTPServer(service)
        client = TestClient(server.app)
        response = client.post('/echo', content=b'Hello Towhee')
        self.assertTrue(isinstance(response.content, bytes))

    def test_no_input(self):
        service = api_service.APIService(desc='test')

        @service.api(path='/no_input')
        def test():
            return 'No input'

        server = HTTPServer(service)
        client = TestClient(server.app)
        response = client.post('/no_input')
        self.assertEqual(from_json(response.content), 'No input')
