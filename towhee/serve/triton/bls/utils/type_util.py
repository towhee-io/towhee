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
import logging

import numpy

from towhee.types import Image, AudioFrame, VideoFrame

logger = logging.getLogger()


NUMPY_TYPES = [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
               numpy.int8, numpy.int16, numpy.int32, numpy.int64,
               numpy.float16, numpy.float32, numpy.float64]


def type_size(towhee_type):
    if towhee_type is Image:
        return 2
    elif towhee_type is VideoFrame:
        return 4
    elif towhee_type is AudioFrame:
        return 4
    else:
        return 1


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
        logger.error('Unsupport type %s', type(towhee_data))
        return None


def client_result_to_op_data(infer_result: 'InferResult', schema: List, name_prefix='OUTPUT'):
    input_count = 0
    outputs = []
    for (towhee_type, _) in schema:
        nps = []
        size = type_size(towhee_type)
        for _ in range(size):
            nps.append(infer_result.as_numpy(name_prefix + str(input_count)))
            input_count += 1

        data = numpy_to_towhee(nps, towhee_type)
        if data is None:
            return None
        outputs.append(data)
    return outputs


def numpy_to_towhee(np_datas, towhee_type):
    if None in np_datas:
        logger.error('Can not convert None to towhee_type')
        return None

    if towhee_type is str:
        return np_datas[0][0].decode('utf-8')
    elif towhee_type is int:
        return int(np_datas[0][0])
    elif towhee_type is float:
        return float(np_datas[0][0])
    elif towhee_type is Image:
        return Image(np_datas[0],
                     np_datas[1][0].decode('utf-8'))
    elif towhee_type is AudioFrame:
        return AudioFrame(np_datas[0],
                          int(np_datas[1][0]),
                          int(np_datas[2][0]),
                          np_datas[3][0].decode('utf-8'))
    elif towhee_type is VideoFrame:
        return VideoFrame(np_datas[0],
                          np_datas[1][0].decode('utf-8'),
                          int(np_datas[2][0]),
                          int(np_datas[3][0]))
    elif towhee_type in NUMPY_TYPES:
        return np_datas[0]
    else:
        logger.error('Unsupport type %s', towhee_type)
        return None


def py_to_np_type(py_data):
    if isinstance(py_data, str):
        return numpy.object_
    elif isinstance(py_data, int):
        return numpy.int64
    elif isinstance(py_data, float):
        return numpy.float64
    elif isinstance(py_data, numpy.ndarray):
        return py_data.dtype
    else:
        logger.error('Unsupport type %s', type(py_data))
        return None
