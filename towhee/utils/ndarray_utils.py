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

from typing import Union
from pathlib import PosixPath
import numpy as np

from towhee.types import Image
from towhee.utils.log import engine_log
try:
    import cv2
except ModuleNotFoundError as e:
    engine_log.error('cv2 not found, you can install via `pip install opencv-python`.')
    raise ModuleNotFoundError('cv2 not found, you can install via `pip install opencv-python`.') from e


def from_src(src: Union[str, PosixPath]) -> 'towhee.types.Image':
    """
    Load the image from url/path as towhee's Image object.

    Args:
        src (`Union[str, path]`):
            The path leads to the image.

    Returns:
        (`towhee.types.Image`)
            The image wrapepd as towhee's Image.
    """
    ndarray_img = cv2.imread(str(src))
    rgb_img = cv2.cvtColor(ndarray_img, cv2.COLOR_BGR2RGB)
    img = from_ndarray(rgb_img, 'RGB')

    return img


def from_ndarray(ndarray_img: np.ndarray, mode: str) -> 'towhee.types.Image':
    """
    Convert an image loaded by cv2 as ndarray into towhee.types.Image.

    Args:
        ndarray_img (`np.ndarray`):
            A image loaed by cv2 as ndarray.

    Returns:
        (`towhee.types.Image`)
            The image wrapepd as towhee Image.
    """
    img_bytes = ndarray_img.tobytes()
    img_width = ndarray_img.shape[1]
    img_height = ndarray_img.shape[0]
    img_channel = len(cv2.split(ndarray_img))
    img_mode = mode
    img_array = ndarray_img

    towhee_img = Image(img_bytes, img_width, img_height, img_channel, img_mode, img_array)

    return towhee_img


def to_ndarray(towhee_img: 'towhee.type.Image') -> np.ndarray:
    """
    Convert a towhee.types.Image into ndarray.

    The mode is same as `towhee_img`, use `towhee_img.mode` to get the information.

    Args:
        towhee_img (`towhee.types.Image`):
            A towhee image.

    Returns:
        (`np.ndarray`)
            The ndarray of the image, the mode is same as the `towhee_img`.
    """
    shape = (towhee_img.height, towhee_img.width, towhee_img.channel)
    data = towhee_img.image

    ndarray_img = np.ndarray(shape, np.uint8, data)

    return ndarray_img


def rgb2bgr(img: Union[np.ndarray, 'towhee.type.Image']) -> np.ndarray:
    """
    Convert an RGB image into ndarray of mode BGR.

    Args:
        img (`Union[np.ndarray, 'towhee.type.Image']`):
            An RGB image either in the form of ndarray or towhee's Image.

    Returns:
        (`np.ndarray`)
            An ndarray in the mode of BGR.
    """
    if isinstance(img, Image):
        if not img.mode.upper() == 'RGB':
            raise ValueError('The input image should be RGB mode.')
        else:
            rgb_img = img.array if isinstance(img.array, np.ndarray) else to_ndarray(img)

    elif isinstance(img, np.ndarray):
        rgb_img = img

    else:
        raise TypeError('Input image should either be an `np.ndarray` or a `towhee.type.Image`.')

    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    return bgr_img
