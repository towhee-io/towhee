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


from typing import List, Dict
from towhee.serve.triton.bls.python_backend_wrapper import pb_utils


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

    def data(self):
        return self._data

    def name(self):
        return self._name


class MockInferResult:
    def __init__(self, response: 'InferenceResponse'):
        self._response = response

    def as_numpy(self, name):
        for ts in self._response.output_tensors():
            if ts.name() == name:
                return ts.as_numpy()
        return None


class CallbackWrapper:
    def __init__(self, callback):
        self._callback = callback

    def __call__(self, response, err):
        if response is None and err is None:
            self._callback(None, None)
        else:
            self._callback(MockInferResult(response), err)


class MockTritonClient:
    '''
    Mock client
    '''
    def __init__(self, models: Dict):
        self._models = models

    def infer(self, model_name: str, inputs: List[MockInferInput]) -> 'InferResult':
        if model_name not in self._models:
            return None
        request = pb_utils.InferenceRequest(
            [pb_utils.Tensor(item.name(), item.data())
             for item in inputs]
        )

        responses = self._models[model_name].execute([request])
        assert len(responses) == 1
        return MockInferResult(responses[0])

    def start_stream(self, callback):
        self._callback = CallbackWrapper(callback)

    def async_stream_infer(self, model_name, inputs: List[MockInferInput]):
        assert self._callback is not None
        request = pb_utils.InferenceRequest(
            [pb_utils.Tensor(item.name(), item.data())
             for item in inputs],
            self._callback
        )

        self._models[model_name].execute([request])

    def stop_stream(self):
        self._callback = None
