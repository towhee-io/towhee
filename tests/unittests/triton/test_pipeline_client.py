# Copyright 2021 Zilliz. All rights reserved.
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
import numpy as np

from towhee.utils.np_format import NumpyArrayEncoder
from towhee.serve.triton.pipeline_client import Client


class MockHttpClient:
    """
    mock http client
    """
    def __init__(self, url):
        self._url = url

    def infer(self, model_name, inputs):
        res = inputs[0].shape()
        return MockRes(model_name, res)

    def close(self):
        pass


class MockRes:
    """
    mock response class for triton client
    """
    def __init__(self, model_name: str, outputs):
        self._name = model_name
        self._outputs = outputs

    def as_numpy(self, name):
        data = json.dumps((self._name, name, self._outputs), cls=NumpyArrayEncoder)
        return np.array([data], dtype=np.object_)


class TestTritonClient(unittest.TestCase):
    """
    Unit test for triton client.
    """
    def test_client(self):
        """
        test http client infer function
        """
        url = '127.0.0.1:8000'
        client = Client(url)
        client._client = MockHttpClient(url)  # pylint:disable=protected-access
        data = 'test.png', np.random.rand(2, 3, 4)
        re1, re2, re3 = client(data)
        self.assertTrue(isinstance(re1, str))
        self.assertEqual(re1, 'pipeline')
        self.assertTrue(isinstance(re2, str))
        self.assertEqual(re2, 'OUTPUT0')
        self.assertTrue(isinstance(re3, list))
        self.assertEqual(re3, [1, 1])

        _, _, re3 = client.batch([['test1.png', np.random.rand(2, 3, 4)], ['test2.png', np.random.rand(2, 3, 4)],
                                  ['test3.png', np.random.rand(2, 3, 4)]])
        self.assertTrue(isinstance(re3, list))
        self.assertEqual(re3, [3, 1])

        client.close()
