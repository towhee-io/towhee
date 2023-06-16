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
import json

from towhee import api_service, pipe
from towhee.serve.http.server import HTTPServer

from fastapi.testclient import TestClient


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

        response = client.post('/echo', json=[1])
        assert response.status_code == 200
        self.assertEqual(json.loads(response.json()), 1)

        response = client.post('/add_one', json=[1])
        assert response.status_code == 200
        self.assertEqual(json.loads(response.json()), 2)


    def test_pipeline(self):
        server = HTTPServer(
            api_service.build_service(
                (pipe.input('nums').map('nums', 'sum', sum).output('sum'), '/sum')
            )
        )

        client = TestClient(server.app)

        response = client.get('/')
        assert response.status_code == 200

        response = client.post('/sum', json=[[1, 2, 3]])
        assert response.status_code == 200
        self.assertEqual(json.loads(response.json())[0][0], 6)
