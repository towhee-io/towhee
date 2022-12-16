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
import struct
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partial

from towhee.dc2 import pipe
from towhee.serve.triton.constant import PIPELINE_NAME
from towhee.serve.triton.pipeline_client import Client, StreamClient, completion_callback
from towhee.utils.tritonclient_utils import InferenceServerException
from towhee.utils.thirdparty.dail_util import dill as pickle
from towhee.utils.np_format import NumpyArrayDecoder, NumpyArrayEncoder
import towhee.serve.triton.bls.pipeline_model as pipe_model
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils

# pylint:disable=protected-access
# pylint:disable=inconsistent-quotes
# pylint:disable=unused-argument


def deserialize_bytes_tensor(encoded_tensor):
    strs = []
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return np.array(strs, dtype=np.object_)


class MockHttpClient:
    """
    mock http client
    """
    def __init__(self, url):
        self._url = url
        self._pipe = None

    def set_pipeline(self, pipeline):
        with TemporaryDirectory(dir='./') as root:
            pf = Path(root) / 'pipe.pickle'
            with open(pf, 'wb') as f:
                pickle.dump(pipeline.dag_repr, f)

            self._pipe = pipe_model.TritonPythonModel()
            self._pipe._load_pipeline(pf)
            self._pipe.pipe.__call__ == self._pipe.pipe.batch  # pylint:disable=pointless-statement

    def infer(self, model_name, inputs):
        in_data_list = deserialize_bytes_tensor(inputs[0]._raw_data)
        json_data = []
        for in_data in in_data_list:
            input_data = json.loads(in_data, cls=NumpyArrayDecoder)
            json_data.append([json.dumps(input_data, cls=NumpyArrayEncoder)])
        triton_input = np.array(json_data, dtype=np.object_)
        inputs = pb_utils.InferenceRequest([pb_utils.Tensor('INPUT0', triton_input)], [], model_name)
        res = self._pipe.execute([inputs])
        return MockRes(res)

    def close(self):
        pass


class MockGrpcClient(StreamClient):
    """
    mock grpc client
    """
    def __init__(self, url: str, model_name: str = PIPELINE_NAME, max_threads: int = 1):
        super().__init__(url, model_name, max_threads)
        self._pipe = None

    def set_pipeline(self, pipeline):
        with TemporaryDirectory(dir='./') as root:
            pf = Path(root) / 'pipe.pickle'
            with open(pf, 'wb') as f:
                pickle.dump(pipeline.dag_repr, f)

            self._pipe = pipe_model.TritonPythonModel()
            self._pipe._load_pipeline(pf)
            self._pipe.pipe.__call__ == self._pipe.pipe.batch  # pylint:disable=pointless-statement

    def stream_client(self, inputs, count):
        callback = partial(completion_callback, self.user_data)
        in_data_list = deserialize_bytes_tensor(inputs[0]._raw_content)
        if len(in_data_list) >= 8:  # max batch size
            callback(None, InferenceServerException('inference request batch-size must be <= 8 for \'pipeline\''))
            return
        json_data = []
        for in_data in in_data_list:
            input_data = json.loads(in_data, cls=NumpyArrayDecoder)
            json_data.append([json.dumps(input_data, cls=NumpyArrayEncoder)])
        triton_input = np.array(json_data, dtype=np.object_)
        inputs = pb_utils.InferenceRequest([pb_utils.Tensor('INPUT0', triton_input)], [], self._model_name)
        outputs = self._pipe.execute([inputs])
        res = MockRes(outputs)
        callback(res, None)


class MockRes:
    """
    mock response class for triton client
    """
    def __init__(self, outputs):
        self._outputs = outputs[0]

    def as_numpy(self, name):
        res = self._outputs.output_tensors()[0].as_numpy()
        return res


class TestTritonClient(unittest.TestCase):
    """
    Unit test for triton client.
    """
    def test_single_params(self):
        p = (
            pipe.input('num')
            .map('num', 'arr', lambda x: x*10)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )

        url = '127.0.0.1:8000'
        with Client(url) as client:
            client._client = MockHttpClient(url)
            client._client.set_pipeline(p)
            res1 = client(1)
            self.assertEqual(len(res1), 1)
            expect1 = [[11]]
            for index, item in enumerate(res1[0], 0):
                self.assertEqual(item, expect1[index])

            # test with batch
            res2 = client.batch([1, 2, 3])
            self.assertEqual(len(res2), 3)
            expect2 = [[11], [22], [33]]
            for index, single in enumerate(res2):
                for item in single:
                    self.assertEqual(item, expect2[index])

    def test_multi_params(self):
        p = (
            pipe.input('nums', 'arr')
            .flat_map('nums', 'num', lambda x: x)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )

        url = '127.0.0.1:8000'
        with Client(url) as client:
            in0 = [1, 2, 3]
            in1 = np.random.rand(3, 3)
            client._client = MockHttpClient(url)
            client._client.set_pipeline(p)
            res1 = client(in0, in1)
            self.assertEqual(len(res1), 1)
            self.assertEqual(len(res1[0]), 3)
            for index, item in enumerate(res1[0], 1):
                self.assertTrue((item[0] == (in1 + index)).all())

            # test with batch
            res2 = client.batch([[in0, in1], [in0, in1], [in0, in1], [in0, in1]])
            self.assertEqual(len(res2), 4)
            for single in res2:
                for index, item in enumerate(single, 1):
                    self.assertTrue((item[0] == (in1 + index)).all())

    def test_stream_single_params(self):
        p = (
            pipe.input('num')
            .map('num', 'arr', lambda x: x*10)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )

        url = '127.0.0.1:8000'
        client = MockGrpcClient(url, max_threads=3)
        client.set_pipeline(p)

        res1 = client(iter([1]))[0]
        self.assertEqual(len(res1), 1)
        expect1 = [[11]]
        for index, item in enumerate(res1[0], 0):
            self.assertEqual(item, expect1[index])

        # test with batch
        res2 = client(iter([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]), 3)[0]
        self.assertEqual(len(res2), 3)
        expect2 = [[11], [22], [33]]
        for index, single in enumerate(res2):
            for item in single:
                self.assertEqual(item, expect2[index])

        # test with error
        with self.assertRaises(Exception):
            _ = client(iter([[1]*10]), 10)

    def test_stream_multi_params(self):
        p = (
            pipe.input('nums', 'arr')
            .flat_map('nums', 'num', lambda x: x)
            .map(('num', 'arr'), 'ret', lambda x, y: x + y)
            .output('ret')
        )

        url = '127.0.0.1:8000'
        client = MockGrpcClient(url)
        client.set_pipeline(p)
        in0 = [1, 2, 3]
        in1 = np.random.rand(3, 3)
        res1 = client(iter([[in0, in1]]))[0]
        self.assertEqual(len(res1), 1)
        self.assertEqual(len(res1[0]), 3)
        for index, item in enumerate(res1[0], 1):
            self.assertTrue((item[0] == (in1 + index)).all())

        # test with batch
        res2 = client(iter([[[in0, in1], [in0, in1]], [[in0, in1], [in0, in1]]]), 2)
        self.assertEqual(len(res2), 2)
        for single in res2:
            single = single[0]
            for index, item in enumerate(single, 1):
                self.assertTrue((item[0] == (in1 + index)).all())
