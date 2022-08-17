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


class MockTritonPythonBackendUtils:
    '''
    mock triton_python_backend_utils, used in UT.
    '''
    @staticmethod
    def get_input_tensor_by_name(r, input_key):
        '''
        Args:
            r: dict
            input_key: str
        return:
            MockTritonPythonBackendTensor
        '''
        return r.get(input_key)

    @staticmethod
    def InferenceResponse(output_tensors, err=None):
        return MockInferenceResponse(output_tensors, err)

    @staticmethod
    def TritonError(msg):
        return MockTritonError(msg)

    @staticmethod
    def Tensor(name: str, data: 'ndarray'):
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
