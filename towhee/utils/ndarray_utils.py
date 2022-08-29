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

import glob
from pathlib import Path, PosixPath
from typing import Union
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

from towhee.types import Image
from towhee.utils.repo_normalize import RepoNormalize
from towhee.utils.cv2_utils import cv2


def from_src(src: Union[str, PosixPath]) -> Image:
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


def from_zip(zip_src: Union[str, Path], pattern: str = '*.JPEG') -> Image:
    """
    Load the image.zip from url/path as towhee's Image object.

    Args:
        zip_src (`Union[str, path]`):
            The path leads to the image.
        pattern (`str`):
            The image pattern to extract.

    Returns:
        (`towhee.types.Image`)
            The image wrapepd as towhee's Image.
    """
    if RepoNormalize(str(zip_src)).url_valid():
        with urlopen(zip_src) as zip_file:
            zip_path = BytesIO(zip_file.read())
    else:
        zip_path = str(Path(zip_src).resolve())
    with ZipFile(zip_path, 'r') as zfile:
        file_list = zfile.namelist()
        path_list = glob.fnmatch.filter(file_list, pattern)
        img_list = []
        for path in path_list:
            data = zfile.read(path)
            ndarray_img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
            rgb_img = cv2.cvtColor(ndarray_img, cv2.COLOR_BGR2RGB)
            img = from_ndarray(rgb_img, 'RGB')
            img_list.append(img)

    return img_list


def from_ndarray(img: np.ndarray, mode: str) -> Image:
    """
    Convert an image loaded by cv2 as ndarray into towhee.types.Image.

    Args:
        img (`np.ndarray`):
            A image loaed by cv2 as ndarray.
        mode (`str`):
            The mode of the image.

    Returns:
        (`towhee.types.Image`)
            The image wrapepd as towhee Image.
    """
    # img_bytes = ndarray_img.tobytes()
    # img_width = ndarray_img.shape[1]
    # img_height = ndarray_img.shape[0]
    # img_channel = len(cv2.split(ndarray_img))

    t_img= Image(img, mode)

    return t_img


def to_ndarray(t_img: Image, dtype=None) -> np.ndarray:
    """
    Convert a towhee.types.Image into ndarray.

    The mode is same as `towhee_img`, use `towhee_img.mode` to get the information.

    Args:
        t_img (`towhee.types.Image`):
            A towhee image.
        dtype:
            The type of the narray.

    Returns:
        (`np.ndarray`)
            The ndarray of the image, the mode is same as the `towhee_img`.
    """
    shape = t_img.shape
    dtype = np.uint8 if not dtype else dtype

    arr = np.ndarray(shape, dtype, t_img)

    return arr


def rgb2bgr(img: Union[np.ndarray, Image]) -> np.ndarray:
    """
    Convert an RGB image into ndarray of mode BGR.

    Args:
        img (`Union[np.ndarray, 'towhee.types.Image']`):
            An RGB image either in the form of ndarray or towhee's Image.

    Returns:
        (`np.ndarray`)
            An ndarray in the mode of BGR.
    """
    if isinstance(img, Image):
        if not img.mode.upper() == 'RGB':
            raise ValueError('The input image should be RGB mode.')
        else:
            rgb_img = to_ndarray(img)

    elif isinstance(img, np.ndarray):
        rgb_img = img

    else:
        raise TypeError('Input image should either be an `np.ndarray` or a `towhee.types.Image`.')

    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    return bgr_img
