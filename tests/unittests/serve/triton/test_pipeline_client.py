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

from typing import List
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from towhee import pipe
from towhee.serve.triton import triton_client
from towhee.utils.thirdparty.dill_util import dill as pickle
import towhee.serve.triton.bls.pipeline_model as pipe_model
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils

# pylint:disable=protected-access
# pylint:disable=inconsistent-quotes
# pylint:disable=unused-argument


class MockInferenceServerClient:
    """
    mock http client
    """
    PIPE = None
    def __init__(self, url):
        self._url = url
        self._pipe = None

    @staticmethod
    def set_pipeline(pipeline):
        with TemporaryDirectory(dir='./') as root:
            pf = Path(root) / 'pipe.pickle'
            with open(pf, 'wb') as f:
                pickle.dump(pipeline.dag_repr, f)

            MockInferenceServerClient._PIPE = pipe_model.TritonPythonModel()
            MockInferenceServerClient._PIPE._load_pipeline(pf)

    async def infer(self, model_name, inputs: List['MockInferInput']):
        inputs = pb_utils.InferenceRequest([pb_utils.Tensor('INPUT0', inputs[0].data())], [], model_name)
        res = MockInferenceServerClient._PIPE.execute([inputs])
        return MockRes(res[0])

    async def close(self):
        pass


class MockInferInput:
    '''
    Mock InferInput
    '''
    def __init__(self, name, shape, datatype):
        self._name = name
        self._shape = shape
        self._datatype = datatype

    def set_data_from_numpy(self, tensor: 'ndarray'):
        self._data = tensor

    def shape(self):
        return self._shape

    def data(self):
        return self._data

    def name(self):
        return self._name


class MockRes:
    """
    mock response class for triton client
    """
    def __init__(self, outputs: '_MockInferenceResponse'):
        self._outputs = outputs

    def as_numpy(self, name):
        return pb_utils.get_output_tensor_by_name(self._outputs, name).as_numpy()


class TestTritonClient(unittest.TestCase):
    """
    Unit test for triton client.
    """
    def setUp(self):
        MockInferenceServerClient.PIPE = None

    @mock.patch('towhee.utils.triton_httpclient.aio_httpclient.InferenceServerClient', new=MockInferenceServerClient)
    @mock.patch('towhee.utils.triton_httpclient.aio_httpclient.InferInput', new=MockInferInput)
    def test_single_params(self):
        p = (
            pipe.input('num')
            .map('num', 'arr', lambda x: x*10)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )
        MockInferenceServerClient.set_pipeline(p)

        url = '127.0.0.1:8000'
        with triton_client.Client(url) as client:
            res1 = client(1)
            self.assertEqual(len(res1), 1)
            expect1 = [11]
            for index, item in enumerate(res1[0], 0):
                self.assertEqual(item, expect1[index])

            res2 = client.batch([1, 2, 3])
            self.assertEqual(len(res2), 3)
            expect2 = [[11], [22], [33]]
            for index, single in enumerate(res2):
                for item in single:
                    self.assertEqual(item, expect2[index])

    @mock.patch('towhee.utils.triton_httpclient.aio_httpclient.InferenceServerClient', new=MockInferenceServerClient)
    @mock.patch('towhee.utils.triton_httpclient.aio_httpclient.InferInput', new=MockInferInput)
    def test_multi_params(self):
        p = (
            pipe.input('nums', 'arr')
            .flat_map('nums', 'num', lambda x: x)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )
        MockInferenceServerClient.set_pipeline(p)
        url = '127.0.0.1:8000'
        with triton_client.Client(url) as client:
            in0 = [1, 2, 3]
            in1 = np.random.rand(3, 3)
            res1 = client(in0, in1)
            self.assertEqual(len(res1), 3)
            for index, item in enumerate(res1, 1):
                self.assertTrue((item[0] == (in1 + index)).all())

            # test with batch
            res2 = client.batch([[in0, in1], [in0, in1], [in0, in1], [in0, in1]])
            self.assertEqual(len(res2), 4)
            for single in res2:
                for index, item in enumerate(single, 1):
                    self.assertTrue((item[0] == (in1 + index)).all())


            # safe batch
            res2 = client.batch([[in0, in1], [in0, in1], [in0, in1], ['err']], batch_size=2, safe=True)
            self.assertIsNone(res2[-1])
            self.assertIsNone(res2[-2])
            self.assertEqual(len(res2), 4)
            for single in res2[:2]:
                for index, item in enumerate(single, 1):
                    self.assertTrue((item[0] == (in1 + index)).all())


            # unsafe batch
            with self.assertRaises(Exception):
                _ = client.batch([[in0, in1], [in0, in1], [in0, in1], [in0, in1], ['err']], batch_size=2, safe=False)
