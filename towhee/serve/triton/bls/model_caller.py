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

from towhee.serve.triton.triton_util import ToTowheeData, ToTritonData


class LocalCaller:
    '''
    Call local models
    '''
    def __init__(self):
        self._client = None

    def gen_inputs(self, datas: List['ndarray'], prefix: str) -> List['InferInput']:
        # ndarray to infers.
        infers = []
        for i in range(len(datas)):
            name = prefix + str(i)
            infer = self._client.InferInput(name, datas[i].shape, np_to_triton_dtype(datas[i]))
            infer.set_data_from_numpy(datas[i])
            infers.append(infer)
        return infers

    def gen_outputs(self, output_schema: List[any]) -> List['InferRequestedOutput']:
        # gen client InferRequestedOutputs
        pass

    def call_model(self, model_name: str, request_data: List[any], output_schema: List[any]) -> any:
        '''
        Args:
            model_name: str
            request_data: List[any]
               towhee_types data

            output_schema: List[any]
               the outputs of model, describe the towhee types.
        '''
        nps = []
        for item in request_data:
            nps.extend(ToTritonData.to_numpy_data(item))
        infers = self.gen_inputs(nps)
        outputs = self.gen_outputs(output_schema)
        response = self._client.infer(model_name, infers, outputs=outputs)
        np_rets = []
        for key in output_keys:
            np_rets.append(response.as_numpy(key))
        return ToTowheeData(np_rets, output_schema)
        

class GrpcCaller:
    def __init__(self, host, port):
        pass

    def call_model(self, model_name: str):
        pass
