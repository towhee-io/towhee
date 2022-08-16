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


import logging

import numpy

from towhee.types import Image, AudioFrame, VideoFrame

logger = logging.getLogger()


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
        logger.error('Unsupport type %s' % type(towhee_data))
        return None
