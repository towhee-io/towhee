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
import queue

from towhee.serve.triton.bls.utils import type_util
from towhee.serve.triton.bls.mock import mock_triton_client as client
from towhee.utils.thirdparty.tritonclient_utils import np_to_triton_dtype


class LocalCallerBase:
    '''
    Local caller base
    '''
    def __init__(self, models):
        self._client = client.MockTritonClient(models)

    def gen_inputs(self, datas: List['ndarray'], prefix: str) -> List['InferInput']:
        infers = []
        for i in range(len(datas)):
            name = prefix + str(i)
            infer = client.MockInferInput(name, datas[i].shape,
                                          np_to_triton_dtype(
                                              type_util.py_to_np_type(datas[i])
                                          )
            )
            infer.set_data_from_numpy(datas[i])
            infers.append(infer)
        return infers


class LocalCaller(LocalCallerBase):
    '''
    Call local models
    '''

    def call_model(self, model_name: str, request_data: List[any], output_schema: List[any]) -> any:
        '''
        Op inputs -> triton_model_inputs -> call_model -> triton_outputs -> op_outputs

        Args:
            model_name: str
            request_data: List[any]
               towhee_types data

            output_schema: List[any]
               the outputs of model, describe the towhee types.
        returns:
            Op results: List
        '''
        # op inputs to triton model inputs
        nps = []
        for item in request_data:
            nps.extend(type_util.to_numpy_data(item))
        infers = self.gen_inputs(nps, 'INPUT')

        # call model
        result = self._client.infer(model_name, infers)

        # triton_outputs to op_outputs
        return type_util.client_result_to_op_data(result, output_schema)


class Callback:
    '''
    Get the respones from model.
    '''
    def __init__(self, q: queue.Queue, schema):
        self.q = q
        self._schema = schema

    def __call__(self, result: 'InferResult', error):
        if result:
            ret = type_util.client_result_to_op_data(result, self._schema)
            self.q.put(ret)
        else:
            self.q.put(None)


class LocalStreamCaller(LocalCallerBase):
    '''
    Stream caller.
    '''
    def start_stream(self, outputs: queue.Queue, output_schema: List[any]):
        self._client.start_stream(callback=Callback(outputs, output_schema))

    def async_stream_call(self, model_name: str, request_data: List) -> any:
        nps = []
        for item in request_data:
            nps.extend(type_util.to_numpy_data(item))
        infers = self.gen_inputs(nps, 'INPUT')

        # call model
        self._client.async_stream_infer(model_name, infers)

    def stop_stream(self):
        self._client.stop_stream()
