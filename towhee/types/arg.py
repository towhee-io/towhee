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

from functools import wraps
from typing import Callable


def arg(arg_idx: int, preprocess: Callable):
    """
    Type utils apply to function arguments.
    Args:
        arg_idx (`int`):
            The index number of the applied argument.
        preprocess (`Callable`):
            The arg preprocess function.
    """
    def decorate(func):
        @ wraps(func)
        def wrapper(*args, **kwargs):
            x = preprocess(args[arg_idx])
            new_args = args[:arg_idx] + (x,) + args[arg_idx + 1:]
            return func(*new_args, **kwargs)
        return wrapper
    return decorate


# pylint: disable=invalid-name
class to_image_color:
    """
    convert images from one color-space to another, like BGR ↔ Gray, BGR ↔ HSV, etc.
    Args:
        mode (`str`):
            The target color space type, for example, 'RGB'.
    Raise:
        ValueError:
            ValueError occurs if the color conversion is not supported.
    """

    def __init__(self, mode: str):
        self._mode = mode

    def __call__(self, img):
        # pylint: disable=import-outside-toplevel
        from towhee.types import image_utils

        return image_utils.to_image_color(img, self._mode)
