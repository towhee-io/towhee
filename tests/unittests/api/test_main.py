# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import asyncio
import time
from unittest import IsolatedAsyncioTestCase
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi.testclient import TestClient
from httpx import AsyncClient
from towhee.serve.api_server.main import app

client = TestClient(app)
pipeline_name = 'pipe'


async def request(method: str, url: str, *args, **kwargs):
    async with AsyncClient(app=app, base_url='http://test') as async_client:
        return await async_client.request(method, url, *args, **kwargs)


def run(method: str, url: str, *args, **kwargs):
    return asyncio.run(request(method, url, *args, **kwargs))


class TestAPI(IsolatedAsyncioTestCase):
    """
    Test api.
    """
    @classmethod
    def setUpClass(cls) -> None:
        response = client.post('/v1/pipeline/create',
                               json={
                                   'dag_json_str': 'test_dag',
                                   'name': pipeline_name,
                                   'description': 'test p0'
                                    })
        cls().assertEqual(response.status_code, 200)
        res = response.json()
        cls().assertEqual(res['status_code'], 0)

    def test_1_create_pipeline(self):
        response = client.post('/v1/pipeline/create',
                               json={
                                   'dag_json_str': 'test_dag',
                                   'name': pipeline_name,
                                   'description': 'test p0'
                               })
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], -1)

    def test_2_update_pipeline(self):
        response = client.post('/v1/pipeline/update',
                               json={
                                   'dag_json_str': 'new_test_dag',
                                   'name': pipeline_name
                                    })
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], 0)

        response = client.post('/v1/pipeline/update',
                               json={
                                   'dag_json_str': 'new_test_dag',
                                   'name': 'none_pipeline_name'
                               })
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], -1)

    def test_3_get_pipelines(self):
        response = client.get('/v1/pipeline/list')
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], 0)
        self.assertEqual(res['data'], {pipeline_name: 'test p0'})

    def test_4_get_pipeline_info(self):
        response = client.get(f'/v1/pipeline/{pipeline_name}/info')
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], 0)
        self.assertEqual(list(res['data'].keys()), ['0', '1'])

        response = client.get('/v1/pipeline/none_pipeline_name/info')
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], -1)

    def test_5_get_pipeline_dag(self):
        response = client.get(f'/v1/pipeline/{pipeline_name}/{1}/dag')
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], 0)
        self.assertEqual(res['data']['dag_str'], 'new_test_dag')

        response = client.get(f'/v1/pipeline/none_pipeline_name/{1}/dag')
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], -1)

    def test_6_delete_pipeline(self):
        response = client.delete('/v1/pipeline/none_pipeline_name')
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res['status_code'], -1)

    async def test_7_async_pipeline(self):
        t1 = time.time()
        client.post('/v1/pipeline/update', json={'dag_json_str': 'new_test_dag_0', 'name': pipeline_name})
        client.get('/v1/pipeline/list')
        client.get(f'/v1/pipeline/{pipeline_name}/info')
        client.get(f'/v1/pipeline/{pipeline_name}/{1}/dag')
        t2 = time.time()

        pools = ThreadPoolExecutor(4)
        task_list = [pools.submit(run, 'POST', '/v1/pipeline/update', json={'dag_json_str': 'new_test_dag_1', 'name': pipeline_name}),
                     pools.submit(run, 'GET', '/v1/pipeline/list'),
                     pools.submit(run, 'GET', f'/v1/pipeline/{pipeline_name}/info'),
                     pools.submit(run, 'GET', f'/v1/pipeline/{pipeline_name}/{1}/dag')]
        t3 = time.time()
        for task in as_completed(task_list):
            task.result()
        t4 = time.time()
        self.assertLess(t4-t3, t2-t1)

    @classmethod
    def tearDownClass(cls):
        response = client.delete(f'/v1/pipeline/{pipeline_name}')
        cls().assertEqual(response.status_code, 200)
        res = response.json()
        cls().assertEqual(res['status_code'], 0)

        os.remove('sql_app.db')
