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

import numpy

from towhee.serve.triton.python_backend_wrapper import pb_utils
from towhee.types import Image, AudioFrame, VideoFrame

logger = logging.getLogger()


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
    def __init__(self, r: 'triton.request', schema: List[Tuple]):
        self._r = r
        self._schema = schema

    def get_towhee_data(self) -> Optional[List[any]]:
        input_count = 0
        outputs = []
        for (towhee_type, shape) in self._schema:
            size = ToTowheeData.type_size(towhee_type)
            data = []
            for _ in range(size):
                input_key = 'INPUT' + str(input_count)
                input_count = input_count + 1
                data.append(pb_utils.get_input_tensor_by_name(self._r, input_key))

            data = ToTowheeData.to_towhee_data(data, towhee_type)
            if data is None:
                return None
            outputs.append(data)
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
            logger.error('Unsupport type %s' % towhee_type)
            return None


class ToTritonData:
    '''
    Convert Towhee data to triton data
    '''
    def __init__(self, towhee_datas):
        self._towhee_datas = towhee_datas

    def to_triton_response(self):
        if self.get_triton_tensor('OUTPUT') is None:
            return None
        return pb_utils.InferenceResponse(output_tensors=self.get_triton_tensor('OUTPUT'))

    def get_triton_tensor(self, name_prefix):
        count = 0
        outputs = []
        for data in self._towhee_datas:
            np_datas = ToTritonData.to_numpy_data(data)
            if np_datas is None:
                return None
            for np_data in np_datas:
                tensor = pb_utils.Tensor(name_prefix + str(count), np_data)
                count += 1
                outputs.append(tensor)
        return outputs

    @staticmethod
    def to_numpy_data(towhee_data):
        if isinstance(towhee_data, Image):
            return [towhee_data.data, numpy.array([towhee_data.mode.encode('utf-8')], dtype=numpy.object_)]
        elif isinstance(towhee_data, VideoFrame):
            return [towhee_data.data,
                    numpy.array([towhee_data.mode.encode('utf-8')], dtype=numpy.object_),
                    numpy.array([towhee_data.timestamp]),
                    numpy.array([towhee_data.key_frame])]
        elif isinstance(towhee_data, AudioFrame):
            return [towhee_data.data,
                    numpy.array([towhee_data.sample_rate]),
                    numpy.array([towhee_data.timestamp]),
                    numpy.array([towhee_data.layout.encode('utf-8')], dtype=numpy.object_)]
        elif isinstance(towhee_data, int):
            return [numpy.array([towhee_data])]
        elif isinstance(towhee_data, float):
            return [numpy.array([towhee_data])]
        elif isinstance(towhee_data, str):
            return [numpy.array([towhee_data.encode('utf-8')], dtype=numpy.object_)]
        elif isinstance(towhee_data, numpy.ndarray):
            return [towhee_data]
        else:
            logger.error('Unsupport type %s' % type(towhee_data))
            return None
