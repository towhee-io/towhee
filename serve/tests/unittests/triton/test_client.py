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
from serve.triton import client

class MockHttpClient():
    """
    mock http client
    """
    def __init__(self, url, model_name='pipeline'):
        self.url = url
        self._model_name = model_name

    def get_model_config(self, model_name):
        res = {}
        res['name'] = model_name
        res['platform'] = 'ensemble'
        res['max_batch_size'] = 1
        res['input'] = []
        input0 = {}
        input0['name'] = 'INPUT0'
        input0['shape'] = [1]
        input0['datatype'] = 'BYTES'
        res['input'].append(input0)
        return res

    def get_model_metadata(self, model_name):
        res = {}
        res['name'] = model_name
        inputs = []
        input0 = {}
        input0['name'] = 'INPUT0'
        input0['datatype'] = 'BYTES'
        input0['shape'] = [1]
        inputs.append(input0)
        res['inputs'] =inputs
        outputs = []
        output0 = {}
        output0['name'] = 'OUTPUT0'
        output0['datatype'] = 'FP32'
        output0['shape'] = [512]
        outputs.append(output0)
        res['outputs'] = outputs
        return res

    def infer(self, model_name, inputs, outputs):
        return MockRes(model_name, inputs, outputs)

class MockGrpcClient():
    """
    mock grpc client
    """
    def __init__(self, url):
        self.url = url

    def get_model_config(self, model_name):
        return GrpcConfig(model_name)

    def get_model_metadata(self, model_name):
        self.name = model_name
        inputs = []
        inputs.append(GrpcMetadataInputs())
        self.inputs = inputs
        outputs = []
        outputs.append(GrpcMetadataOutputs())
        self.outputs = outputs
        return self

    def infer(self, model_name, inputs, outputs):
        return MockRes(model_name, inputs, outputs)

class GrpcConfig():
    """
    grpc config class
    """
    def __init__(self, model_name):
        self.config = GrpcConfigConfig(model_name)

class GrpcConfigConfig():
    """
    grpc config adapt to triton grpc client
    """
    def __init__(self, model_name):
        self.name = model_name
        self.max_batch_size = 0

class GrpcMetadataInputs():
    """
    grpc metadata inputs
    """
    def __init__(self):
        self.name = 'INPUT0'
        self.datatype = 'BYTES'
        self.shape = [1]

class GrpcMetadataOutputs():
    """
    grpc metadata outputs
    """
    def __init__(self):
        self.name = 'OUTPUT0'
        self.datatype = 'FP32'
        self.shape = [512]

class MockRes:
    """
    mock response class for triton client
    """
    def __init__(self, model_name, inputs, outputs):
        self.name = model_name
        self.inputs = inputs
        self.outputs = outputs

    def as_numpy(self, name):
        _ = name
        return [1, 2, 3]

class TestTritonClient(unittest.TestCase):
    """
    Unit test for triton client.
    """
    def test_http_client(self):
        tclient = client.Client('127.0.0.1:8001', protocol='http')
        tclient.client = MockHttpClient('127.0.0.1:80001')
        res = tclient.serve('/test.jpg')

        expect = {}
        expect['OUTPUT0'] = [1, 2, 3]
        self.assertEqual(res, expect)

    def test_grpc_client(self):
        tclient = client.Client('127.0.0.1:8002')
        tclient.client = MockGrpcClient('127.0.0.1:8002')
        res = tclient.serve('/test.jpg')
        expect = {}
        expect['OUTPUT0'] = [1, 2, 3]
        self.assertEqual(res, expect)
