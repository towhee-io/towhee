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


class MockTritonPythonBackendUtils:
    '''
    mock triton_python_backend_utils, used in UT.
    '''

    TRITONSERVER_RESPONSE_COMPLETE_FINAL = 1

    @staticmethod
    def get_input_tensor_by_name(r, input_key):
        '''
        Args:
            r: MockInferenceRequest
            input_key: str
        return:
            MockTritonPythonBackendTensor
        '''
        for item in r.inputs():
            if item.name() == input_key:
                return item
        return None

    @staticmethod
    def InferenceResponse(output_tensors: List['MockTritonPythonBackendTensor'], err=None):  # pylint: disable=invalid-name
        return MockInferenceResponse(output_tensors, err)

    @staticmethod
    def InferenceRequest(inputs: 'MockTritonPythonBackendTensor'):  # pylint: disable=invalid-name
        return MockInferenceRequest(inputs)

    @staticmethod
    def TritonError(msg):  # pylint: disable=invalid-name
        return MockTritonError(msg)

    @staticmethod
    def Tensor(name: str, data: 'ndarray'):  # pylint: disable=invalid-name
        return MockTritonPythonBackendTensor(name, data)


class MockTritonPythonBackendTensor:
    '''
    Mock python_backend tensor object.
    '''
    def __init__(self, name:str, data: 'ndarray'):
        self._name = name
        self._data = data

    def name(self):
        return self._name

    def as_numpy(self):
        return self._data


class MockResponseSender:
    def __init__(self, callback):
        self._callback = callback

    def send(self, response=None, flags=0):
        if flags != MockTritonPythonBackendUtils.TRITONSERVER_RESPONSE_COMPLETE_FINAL:
            self._callback(response, None)
        else:
            self._callback(None, None)


class MockInferenceRequest:
    def __init__(self, tensors: MockTritonPythonBackendTensor, callback=None):
        self._tensors = tensors
        self._sender = MockResponseSender(callback)

    def inputs(self):
        return self._tensors

    def get_response_sender(self):
        return self._sender


class MockInferenceResponse:
    def __init__(self, tensors, err):
        self._tensors = tensors
        self._err = err

    def output_tensors(self):
        return self._tensors

    def has_error(self):
        return self._err is not None


class MockTritonError:
    def __init__(self, msg):
        self._msg = msg

    def message(self):
        return self._msg
