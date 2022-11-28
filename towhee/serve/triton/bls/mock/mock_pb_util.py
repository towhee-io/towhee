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

# pylint: skip-file

from typing import List


class _MockTritonPythonBackendTensor:
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

class _MockInferenceResponseSender:
    # def __init__(self, callback):
    #     self._callback = callback

    def send(self, response=None, flags=0):
        # if flags != MockTritonPythonBackendUtils.TRITONSERVER_RESPONSE_COMPLETE_FINAL:
        #     self._callback(response, None)
        # else:
        #     self._callback(None, None)
        pass

class _MockInferenceRequest:
    def __init__(self, inputs: _MockTritonPythonBackendTensor,
                 requested_output_names: List[str],
                 model_name: str):
        self._tensors = inputs
        self._requested_output_names = requested_output_names
        self._model_name = model_name
        # self._sender = _MockResponseSender(callback)

    def inputs(self):
        return self._tensors

    def exec(self):
        pass

    def requested_output_names(self):
        pass

    def get_response_sender(self):
        pass


class _MockInferenceResponse:
    def __init__(self, output_tensors, error=None):
        self._tensors = output_tensors
        self._err = error

    def output_tensors(self):
        return self._tensors

    def has_error(self):
        return self._err is not None

    def error(self):
        return self._err


class _MockTritonError:
    def __init__(self, msg):
        self._msg = msg

    def message(self):
        return self._msg


class MockTritonPythonBackendUtils:
    '''
    mock triton_python_backend_utils, used in UT.
    '''

    TRITONSERVER_RESPONSE_COMPLETE_FINAL = 1

    InferenceResponse = _MockInferenceResponse
    InferenceRequest = _MockInferenceRequest
    TritonError = _MockTritonError
    Tensor = _MockTritonPythonBackendTensor

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







