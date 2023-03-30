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
import unittest
import os

from towhee.utils.thirdparty.fastapi_utils import TestClient
from towhee.serve.api.main import app

client = TestClient(app)
pipeline_name = 'pipe0'


class TestAPI(unittest.TestCase):
    """
    Test api.
    """
    def test_1_create_pipeline(self):
        response = client.post('/pipeline/create',
                               json={
                                   'dag_json_str': 'test_dag',
                                   'name': pipeline_name,
                                   'description': 'test p0'
                                    })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 200, 'msg': 'Successfully create pipeline.'})

    def test_2_update_pipeline(self):
        response = client.post('/pipeline/update',
                               json={
                                   'dag_json_str': 'new_test_dag',
                                   'name': pipeline_name
                                    })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 200, 'msg': 'Successfully update pipeline.'})

    def test_3_get_pipelines(self):
        response = client.get('/pipelines')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 200, 'msg': {pipeline_name: 'test p0'}})

    def test_4_get_pipeline_info(self):
        response = client.get(f'/{pipeline_name}/info')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(response.json()['msg'].keys()), ['0', '1'])

    def test_5_get_pipeline_dag(self):
        response = client.get(f'/{pipeline_name}/{1}/dag')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 200, 'msg': 'new_test_dag'})

    @classmethod
    def tearDownClass(cls):
        response = client.delete(f'/{pipeline_name}')
        cls().assertEqual(response.status_code, 200)
        cls().assertEqual(response.json(), {'status': 200, 'msg': 'Successfully deleted the pipeline.'})

        os.remove('sql_app.db')
