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

from collections import namedtuple
from typing import List, Tuple, Dict, Any, Callable, get_args, get_origin

import towhee
import numpy


def handle_type_annotations(type_annotations: List[Tuple[Any, Tuple]], callbacks: Dict[Any, Callable]):
    """
    Template for handling type annotations

    Args:
        type_annotations (List[Tuple[Any, Tuple]]): a list of type annotations,
        each annotation corresponds to an input or output, containing a pair of
        type class and shape. The supported type classes are typing.List,
        numpy numerical types (numpy.uint8, numpy.uint16, numpy.uint32,
        numpy.uint64, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
        numpy.float16, numpy.float32, numpy.float64),
        python numerical types (str, bool, int, float),
        towhee.types.Image, towhee.types.AudioFrame, towhee.types.VideoFrame.

        callbacks (Dict[Any, Callable]): a dict of callback functions.

    Returns:
        A list of generated codes.
    """

    results = []

    for annotation in type_annotations:
        t, shape = annotation
        is_list = False
        if get_origin(t) is list:
            # we treat list as additional dim
            t, = get_args(t)
            is_list = True

        results.append(callbacks[t](t, shape, is_list))

    return results


def tensor2str(placehoder: str):
    return 'str(' + placehoder + '.as_numpy()).decode(\'utf-8\')'


def tensor2bool(placehoder: str):
    return 'bool(' + placehoder + '.as_numpy())'


def tensor2int(placehoder: str):
    return 'int(' + placehoder + '.as_numpy())'


def tensor2float(placehoder: str):
    return 'float(' + placehoder + '.as_numpy())'


def tensor2ndarray(placehoder: str):
    return placehoder + '.as_numpy()'


AttrInfo = namedtuple('AttrInfo', ['tensor_placeholder', 'obj_placeholder', 'shape', 'triton_dtype', 'numpy_dtype'])


class TypeInfo:
    def __init__(self, attr_info, t, shape, is_list=False):
        self.attr_info = attr_info
        self.type = t
        self.shape = shape
        self.is_list = is_list


class ImageType:
    """
    A collection of type handling function of `towhee.types.Image`
    """

    @staticmethod
    def type_info(t, shape, is_list):
        data_part = AttrInfo('$t_data', '$obj', shape, 'TYPE_INT8', 'numpy.int8')
        mode_part = AttrInfo('$t_mode', '$obj.mode', (), 'TYPE_STRING', 'numpy.object_')

        return TypeInfo([data_part, mode_part], t, shape, is_list)

    # pylint: disable=unused-argument
    @staticmethod
    def init_code(
            shape,
            data_placehoder='$t_data',
            mode_placehoder='$t_mode'
    ):
        init_args = [
            tensor2ndarray(data_placehoder),
            tensor2str(mode_placehoder)
        ]
        return 'towhee._types.Image(' + ', '.join(init_args) + ')'


class VideoFrameType:
    """
    A collection of type handling function of `towhee.types.VideoFrame`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        data_part = AttrInfo('$t_data', '$obj', shape, 'TYPE_INT8', 'numpy.int8')
        mode_part = AttrInfo('$t_mode', '$obj.mode', (-1, ), 'TYPE_STRING', 'numpy.object_')
        timestamp_part = AttrInfo('$t_timestamp', '$obj.timestamp', (), 'TYPE_INT64', 'numpy.int64')
        key_frame_part = AttrInfo('$t_key_frame', '$obj.key_frame', (), 'TYPE_INT8', 'numpy.int8')

        return TypeInfo([data_part, mode_part, timestamp_part, key_frame_part], t, shape, is_list)

    # pylint: disable=unused-argument
    @staticmethod
    def init_code(
            shape,
            data_placeholder='$t_data',
            mode_placeholder='$t_mode',
            timestamp_placeholder='$t_timestamp',
            key_frame_placeholder='$t_key_frame'
    ):
        init_args = [
            tensor2ndarray(data_placeholder),
            tensor2str(mode_placeholder),
            tensor2int(timestamp_placeholder),
            tensor2int(key_frame_placeholder)
        ]
        return 'towhee.types.VideoFrame(' + ', '.join(init_args) + ')'


class AudioFrameType:
    """
    A collection of type handling function of `towhee.types.AudioFrame`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        data_part = AttrInfo('$t_data', '$obj', shape, 'TYPE_INT32', 'numpy.int32')
        sample_rate_part = AttrInfo('$t_sample_rate', '$obj.sample_rate', (), 'TYPE_INT32', 'numpy.int32')
        timestamp_part = AttrInfo('$obj.timestamp', (), 'TYPE_INT64', 'numpy.int64')
        layout_part = AttrInfo('$t_layout', '$obj.layout', (-1, ), 'TYPE_STRING', 'numpy.object_')

        return TypeInfo([data_part, sample_rate_part, timestamp_part, layout_part], t, shape, is_list)

    # pylint: disable=unused-argument
    @staticmethod
    def init_code(
        shape,
        data_placeholder='$t_data',
        sample_rate_placeholder='$t_sample_rate',
        timestamp_placeholder='$t_timestamp',
        layout_placeholder='$t_layout'
    ):
        init_args = [
            tensor2ndarray('$t_data'),
            tensor2int('$t_sample_rate'),
            tensor2int('$t_timestamp'),
            tensor2str('$t_layout')
        ]
        return 'towhee.types.AudioFrame(' + ', '.join(init_args) + ')'


class NpUint8Type:
    """
    A collection of type handling function of `numpy.uint8`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_UINT8', 'numpy.uint8')]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2int(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpUint16Type:
    """
    A collection of type handling function of `numpy.uint16`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_UINT16', 'numpy.uint16', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2int(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpUint32Type:
    """
    A collection of type handling function of `numpy.uint32`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_UINT32', 'numpy.uint32', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2int(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpUint64Type:
    """
    A collection of type handling function of `numpy.uint64`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_UINT64', 'numpy.uint64', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2int(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpInt8Type:
    """
    A collection of type handling function of `numpy.int8`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_INT8', 'numpy.int8', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2int(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpInt16Type:
    """
    A collection of type handling function of `numpy.int16`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_INT16', 'numpy.int16', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2int(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpInt32Type:
    """
    A collection of type handling function of `numpy.int32`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_INT32', 'numpy.int32', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2int(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpInt64Type:
    """
    A collection of type handling function of `numpy.int64`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_INT64', 'numpy.int64', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2int(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpFloat16Type:
    """
    A collection of type handling function of `numpy.float16`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_FP16', 'numpy.float16', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2float(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpFloat32Type:
    """
    A collection of type handling function of `numpy.float32`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_FP32', 'numpy.float32', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2float(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class NpFloat64Type:
    """
    A collection of type handling function of `numpy.float64`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_FP64', 'numpy.float64', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        if len(shape) == 0:
            return tensor2float(data_placeholder)
        else:
            return tensor2ndarray(data_placeholder)


class IntType:
    """
    A collection of type handling function of `int`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_INT64', 'numpy.int64', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    # pylint: disable=unused-argument
    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        return tensor2int(data_placeholder)


class FloatType:
    """
    A collection of type handling function of `float`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_FP64', 'numpy.float64', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    # pylint: disable=unused-argument
    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        return tensor2float(data_placeholder)


class BoolType:
    """
    A collection of type handling function of `bool`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_BOOL', 'bool', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    # pylint: disable=unused-argument
    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        return tensor2bool(data_placeholder)


class StringType:
    """
    A collection of type handling function of `str`
    """

    @ staticmethod
    def type_info(t, shape, is_list):
        attr_info = [AttrInfo('$t_data', '$obj', shape, 'TYPE_STRING', 'numpy.object_', is_list)]
        return TypeInfo(attr_info, t, shape, is_list)

    # pylint: disable=unused-argument
    @staticmethod
    def init_code(shape, data_placeholder='$t_data'):
        return tensor2str(data_placeholder)


def get_type_info(type_annotations: List[Tuple[Any, Tuple]]) -> List[TypeInfo]:
    # pylint: disable=protected-access
    callbacks = {
        towhee._types.Image: ImageType.type_info,
        towhee.types.VideoFrame: VideoFrameType.type_info,
        towhee.types.AudioFrame: AudioFrameType.type_info,
        numpy.uint8: NpUint8Type.type_info,
        numpy.uint16: NpUint16Type.type_info,
        numpy.uint32: NpUint32Type.type_info,
        numpy.uint64: NpUint64Type.type_info,
        numpy.int8: NpInt8Type.type_info,
        numpy.int16: NpInt16Type.type_info,
        numpy.int32: NpInt32Type.type_info,
        numpy.int64: NpInt64Type.type_info,
        numpy.float16: NpFloat16Type.type_info,
        numpy.float32: NpFloat32Type.type_info,
        numpy.float64: NpFloat64Type.type_info,
        bool: BoolType.type_info,
        int: IntType.type_info,
        float: FloatType.type_info,
        str: StringType.type_info,
    }

    return handle_type_annotations(type_annotations, callbacks)
