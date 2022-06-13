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

from towhee.types import Image, AudioFrame, VideoFrame
from typing import List, Tuple, Dict, Any, Callable, get_args, get_origin


def gen_from_type_annotations(type_annotations: List[Tuple[Any, Tuple]], codegen_funcs: Dict[Any, Callable]):
    """
    Generate inputs/outputs definitions from type hints

    Args:
        type_annotations (List[Tuple[Any, Tuple]]): a list of type annotations,
        each annotation corresponds to an input or output, containing a pair of
        type class and shape. The supported type classes are typing.List, numpy.ndarray,
        str, bool, int, float, towhee.types.Image, towhee.types.AudioFrame, towhee.types.VideoFrame.

        gen_funcs (Dict[Any, Callable]): a dict of codegen functions against each type alias.

    Returns:
        A list of generated codes.
    """

    results = []

    for annotation in type_annotations:
        t, shape = annotation
        if get_origin(t) is list:
            # we treat list as additional dim
            t, = get_args(t)
            shape = (-1,) + shape

        results.append(codegen_funcs[t](t, shape))

    return results
