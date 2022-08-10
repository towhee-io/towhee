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


import numpy

from towhee.serve.triton.python_backend_wrapper import pb_utils
from towhee.types import Image, AudioFrame, VideoFrame


NUMPY_TYPES = [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
               numpy.int8, numpy.int16, numpy.int32, numpy.int64,
               numpy.float16, numpy.float32, numpy.float64]


class ToTowheeData:
    '''
    Read data from triton requests and convert to towhee types
    args:
        r:  triton python_backend request
        schema: op input schema
    '''
    def __init__(self, r, schema):
        self._r = r
        self._schema = schema

    def get_towhee_data(self):
        input_count = 0
        outputs = []
        for (towhee_type, shape) in self._schema:
            size = ToTowheeData.type_size(towhee_type)
            data = []
            for _ in range(size):
                input_key = 'INPUT' + str(input_count)
                input_count = input_count + 1
                data.append(pb_utils.get_input_tensor_by_name(self._r, input_key))
            outputs.append(ToTowheeData.to_towhee_data(data, towhee_type))
        return outputs

    @staticmethod
    def type_size(towhee_type):
        if towhee_type is Image:
            return 2
        elif towhee_type is VideoFrame:
            return 4
        elif towhee_type is AudioFrame:
            return 4
        else:
            return 1

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
        elif towhee_type in NUMPY_TYPES:
            return triton_data[0].as_numpy()
        else:
            return None