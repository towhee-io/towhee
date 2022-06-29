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
import serve.triton.client as client

model_name = 'pipeline'
class mockHttpClient():
    def __init__(self, url):
        self.url = url
        self._model_name = model_name
    
    def get_model_config(self, model_name):
        res = dict()
        res['name'] = model_name
        res['platform'] = 'ensemble'
        res['max_batch_size'] = 1
        res['input'] = []
        input0 = dict()
        input0['name'] = 'INPUT0'
        input0['shape'] = [1]
        input0['datatype'] = 'BYTES'
        res['input'].append(input0)
        return res

    def get_model_metadata(self, model_name):
        res = dict()
        res['name'] = model_name
        inputs = []
        input0 = dict()
        input0['name'] = 'INPUT0'
        input0['datatype'] = 'BYTES'
        input0['shape'] = [1]
        inputs.append(input0)
        res['inputs'] =inputs
        outputs = []
        output0 = dict()
        output0['name'] = 'OUTPUT0'
        output0['datatype'] = 'FP32'
        output0['shape'] = [512]
        outputs.append(output0)
        res['outputs'] = outputs
        return res

    def infer(self, model_name, inputs, outputs):
        return mockRes(model_name, inputs, outputs)

class mockGrpcClient():
    def __init__(self, url):
        self.url = url
        self.model_name = model_name
        
    def get_model_config(self, model_name):
        return grpcConfig(model_name)
    
    def get_model_metadata(self, model_name):
        inputs = []
        inputs.append(grpcMetadataInputs())
        self.inputs = inputs
        outputs = []
        outputs.append(grpcMetadataOutputs())
        self.outputs = outputs
        return self
    
    def infer(self, model_name, inputs, outputs):
        return mockRes(model_name, inputs, outputs)

class grpcConfig():
    def __init__(self, model_name):
        self.config = grpcConfigConfig(model_name)

class grpcConfigConfig():
    def __init__(self, model_name):
        self.name = model_name
        self.max_batch_size = 0

class grpcMetadataInputs():
    def __init__(self):
        self.name = 'INPUT0'
        self.datatype = 'BYTES'
        self.shape = [1]

class grpcMetadataOutputs():
    def __init__(self):
        self.name = 'OUTPUT0'
        self.datatype = 'FP32'
        self.shape = [512]

class mockRes:
    def __init__(self, model_name, inputs, outputs):
        self.name = model_name
        self.inputs = inputs
        self.outputs = outputs

    def as_numpy(self, name):
        return [1, 2, 3]

class TestTritonClient(unittest.TestCase):
    """
    Unit test for triton client.
    """
    

    def test_http_client(self):
        tclient = client.Client('http', '127.0.0.1:8001')
        tclient.client = mockHttpClient('127.0.0.1:80001')
        res = tclient.serve('/test.jpg')

        expect = dict()
        expect['OUTPUT0'] = [1, 2, 3]
        self.assertEqual(res, expect)
    
    def test_grpc_client(self):
        tclient = client.Client('grpc', '127.0.0.1:8002')
        tclient.client = mockGrpcClient('127.0.0.1:8002')
        res = tclient.serve('/test.jpg')
        expect = dict()
        expect['OUTPUT0'] = [1, 2, 3]
        self.assertEqual(res, expect)


    