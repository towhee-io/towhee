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

from typing import List, Tuple, Dict, Any, Callable, get_args, get_origin


def handle_type_annotations(type_annotations: List[Tuple[Any, Tuple]], callbacks: Dict[Any, Callable]):
    """
    Template for handling type annotations

    Args:
        type_annotations (List[Tuple[Any, Tuple]]): a list of type annotations,
        each annotation corresponds to an input or output, containing a pair of
        type class and shape. The supported type classes are typing.List,
        numpy numerical types (numpy.bool_, numpy.uint8, numpy.uint16, numpy.uint32,
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


class ImageType:
    """
    A collection of type handling function of `towhee.types.Image`
    """

    @staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            data_part = ('$obj', (-1, ) + shape, 'TYPE_INT8', is_list)
            mode_part = ('$obj.mode', (-1, ), 'TYPE_STRING', is_list)
        else:
            data_part = ('$obj', shape, 'TYPE_INT8', is_list)
            mode_part = ('$obj.mode', (), 'TYPE_STRING', is_list)

        return [data_part, mode_part]

    # pylint: disable=unused-argument
    @staticmethod
    def init_obj(shape):
        init_args = [
            tensor2ndarray('$t_data'),
            tensor2str('$t_mode')
        ]
        return 'towhee._types.Image(' + ','.join(init_args) + ')'


class VideoFrameType:
    """
    A collection of type handling function of `towhee.types.VideoFrame`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            data_part = ('$obj', (-1, ) + shape, 'TYPE_INT8', is_list)
            mode_part = ('$obj.mode', (-1, -1), 'TYPE_STRING', is_list)
            timestamp_part = ('$obj.timestamp', (-1, ), 'TYPE_INT64', is_list)
            key_frame_part = ('$obj.key_frame', (-1, ), 'TYPE_INT8', is_list)
        else:
            data_part = ('$obj', shape, 'TYPE_INT8', is_list)
            mode_part = ('$obj.mode', (-1, ), 'TYPE_STRING', is_list)
            timestamp_part = ('$obj.timestamp', (), 'TYPE_INT64', is_list)
            key_frame_part = ('$obj.key_frame', (), 'TYPE_INT8', is_list)

        return [data_part, mode_part, timestamp_part, key_frame_part]

    # pylint: disable=unused-argument
    @staticmethod
    def init_obj(shape):
        init_args = [
            tensor2ndarray('$t_data'),
            tensor2str('$t_mode'),
            tensor2int('$t_timestamp'),
            tensor2int('$t_key_frame')
        ]
        return 'towhee.types.VideoFrame(' + ','.join(init_args) + ')'


class AudioFrameType:
    """
    A collection of type handling function of `towhee.types.AudioFrame`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            data_part = ('$obj', (-1, ) + shape, 'TYPE_INT32', is_list)
            sample_rate_part = ('$obj.sample_rate', (-1, ), 'TYPE_INT32', is_list)
            timestamp_part = ('$obj.timestamp', (-1, ), 'TYPE_INT64', is_list)
            layout_part = ('$obj.layout', (-1, -1), 'TYPE_STRING', is_list)
        else:
            data_part = ('$obj', shape, 'TYPE_INT32', is_list)
            sample_rate_part = ('$obj.sample_rate', (), 'TYPE_INT32', is_list)
            timestamp_part = ('$obj.timestamp', (), 'TYPE_INT64', is_list)
            layout_part = ('$obj.layout', (-1, ), 'TYPE_STRING', is_list)

        return [data_part, sample_rate_part, timestamp_part, layout_part]

    # pylint: disable=unused-argument
    @staticmethod
    def init_obj(shape):
        init_args = [
            tensor2ndarray('$t_data'),
            tensor2int('$t_sample_rate'),
            tensor2int('$t_timestamp'),
            tensor2str('$t_layout')
        ]
        return 'towhee.types.AudioFrame(' + ','.join(init_args) + ')'


class NpBoolType:
    """
    A collection of type handling function of `numpy.bool_`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_BOOL', is_list)]
        else:
            return [('$obj', shape, 'TYPE_BOOL', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2bool('$t')
        else:
            return tensor2ndarray('$t')


class NpUint8Type:
    """
    A collection of type handling function of `numpy.uint8`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_UINT8', is_list)]
        else:
            return [('$obj', shape, 'TYPE_UINT8', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2int('$t')
        else:
            return tensor2ndarray('$t')


class NpUint16Type:
    """
    A collection of type handling function of `numpy.uint16`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_UINT16', is_list)]
        else:
            return [('$obj', shape, 'TYPE_UINT16', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2int('$t')
        else:
            return tensor2ndarray('$t')


class NpUint32Type:
    """
    A collection of type handling function of `numpy.uint32`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_UINT32', is_list)]
        else:
            return [('$obj', shape, 'TYPE_UINT32', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2int('$t')
        else:
            return tensor2ndarray('$t')


class NpUint64Type:
    """
    A collection of type handling function of `numpy.uint64`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_UINT64', is_list)]
        else:
            return [('$obj', shape, 'TYPE_UINT64', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2int('$t')
        else:
            return tensor2ndarray('$t')


class NpInt8Type:
    """
    A collection of type handling function of `numpy.int8`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_INT8', is_list)]
        else:
            return [('$obj', shape, 'TYPE_INT8', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2int('$t')
        else:
            return tensor2ndarray('$t')


class NpInt16Type:
    """
    A collection of type handling function of `numpy.int16`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_INT16', is_list)]
        else:
            return [('$obj', shape, 'TYPE_INT16', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2int('$t')
        else:
            return tensor2ndarray('$t')


class NpInt32Type:
    """
    A collection of type handling function of `numpy.int32`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_INT32', is_list)]
        else:
            return [('$obj', shape, 'TYPE_INT32', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2int('$t')
        else:
            return tensor2ndarray('$t')


class NpInt64Type:
    """
    A collection of type handling function of `numpy.int64`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_INT64', is_list)]
        else:
            return [('$obj', shape, 'TYPE_INT64', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2int('$t')
        else:
            return tensor2ndarray('$t')


class NpFloat16Type:
    """
    A collection of type handling function of `numpy.float16`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_FP16', is_list)]
        else:
            return [('$obj', shape, 'TYPE_FP16', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2float('$t')
        else:
            return tensor2ndarray('$t')


class NpFloat32Type:
    """
    A collection of type handling function of `numpy.float32`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_FP32', is_list)]
        else:
            return [('$obj', shape, 'TYPE_FP32', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2float('$t')
        else:
            return tensor2ndarray('$t')


class NpFloat64Type:
    """
    A collection of type handling function of `numpy.float64`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_FP64', is_list)]
        else:
            return [('$obj', shape, 'TYPE_FP64', is_list)]

    @staticmethod
    def init_obj(shape):
        if len(shape) == 0:
            return tensor2float('$t')
        else:
            return tensor2ndarray('$t')


class IntType:
    """
    A collection of type handling function of `int`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_INT64', is_list)]
        else:
            return [('$obj', shape, 'TYPE_INT64', is_list)]

    # pylint: disable=unused-argument
    @staticmethod
    def init_obj(shape):
        return tensor2int('$t')


class FloatType:
    """
    A collection of type handling function of `float`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_FP64', is_list)]
        else:
            return [('$obj', shape, 'TYPE_FP64', is_list)]

    # pylint: disable=unused-argument
    @staticmethod
    def init_obj(shape):
        return tensor2float('$t')


class BoolType:
    """
    A collection of type handling function of `bool`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_BOOL', is_list)]
        else:
            return [('$obj', shape, 'TYPE_BOOL', is_list)]

    # pylint: disable=unused-argument
    @staticmethod
    def init_obj(shape):
        return tensor2bool('$t')


class StringType:
    """
    A collection of type handling function of `str`
    """

    @ staticmethod
    def unpack_attrs(shape, is_list):
        if is_list:
            return [('$obj', (-1, ) + shape, 'TYPE_STRING', is_list)]
        else:
            return [('$obj', shape, 'TYPE_STRING', is_list)]

    # pylint: disable=unused-argument
    @staticmethod
    def init_obj(shape):
        return tensor2str('$t')
