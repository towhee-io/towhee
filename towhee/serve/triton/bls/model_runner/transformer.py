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


from typing import List, Optional, Tuple
import logging

from towhee.serve.triton.bls.python_backend_wrapper import pb_utils
from towhee.types import Image, AudioFrame, VideoFrame
from towhee.serve.triton.bls.utils import type_util

logger = logging.getLogger()


class RequestToOpInputs:
    '''
    Read data from triton requests and convert to towhee types
    args:
        r:  triton python_backend request
        schema: op input schema
    '''
    def __init__(self, request: 'triton.request', schema: List[Tuple]):
        self._request = request
        self._schema = schema

    def get_towhee_data(self) -> Optional[List[any]]:
        input_count = 0
        outputs = []
        for (towhee_type, _) in self._schema:
            size = type_util.type_size(towhee_type)
            data = []
            for _ in range(size):
                input_key = 'INPUT' + str(input_count)
                input_count = input_count + 1
                data.append(pb_utils.get_input_tensor_by_name(self._request, input_key))

            data = RequestToOpInputs.to_towhee_data(data, towhee_type)
            if data is None:
                return None
            outputs.append(data)
        return outputs

    @staticmethod
    def to_towhee_data(triton_data, towhee_type):
        if towhee_type is str:
            return triton_data[0].as_numpy()[0].decode('utf-8')
        elif towhee_type is int:
            return int(triton_data[0].as_numpy()[0])
        elif towhee_type is float:
            return float(triton_data[0].as_numpy()[0])
        elif towhee_type is Image:
            return Image(triton_data[0].as_numpy(),
                         triton_data[1].as_numpy()[0].decode('utf-8'))
        elif towhee_type is AudioFrame:
            return AudioFrame(triton_data[0].as_numpy(),
                              int(triton_data[1].as_numpy()[0]),
                              int(triton_data[2].as_numpy()[0]),
                              triton_data[3].as_numpy()[0].decode('utf-8'))
        elif towhee_type is VideoFrame:
            return VideoFrame(triton_data[0].as_numpy(),
                              triton_data[1].as_numpy()[0].decode('utf-8'),
                              int(triton_data[2].as_numpy()[0]),
                              int(triton_data[3].as_numpy()[0]))
        elif towhee_type in type_util.NUMPY_TYPES:
            return triton_data[0].as_numpy()
        else:
            logger.error('Unsupport type %s', towhee_type)
            return None


class OpOutputToResponses:
    '''
    Convert Op output data to triton response
    '''
    def __init__(self, towhee_datas):
        self._towhee_datas = towhee_datas

    def to_triton_responses(self):
        tensors = self.get_triton_tensor()
        if tensors is None:
            return pb_utils.InferenceResponse([], err=pb_utils.TritonError('Gen respones failed'))
        return pb_utils.InferenceResponse(output_tensors=tensors)

    def get_triton_tensor(self, name_prefix='OUTPUT'):
        count = 0
        outputs = []
        for data in self._towhee_datas:
            np_datas = type_util.to_numpy_data(data)
            if np_datas is None:
                return None
            for np_data in np_datas:
                tensor = pb_utils.Tensor(name_prefix + str(count), np_data)
                count += 1
                outputs.append(tensor)
        return outputs
