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

import io
import zipfile
import glob
from pathlib import Path
from typing import Union
from pathlib import PosixPath
import numpy as np

from towhee.utils.log import engine_log
from towhee.types import Image

try:
    from PIL import Image as PILImage
except ModuleNotFoundError as e:
    engine_log.error('PIL not found, you can install via `pip install pillow`.')
    raise ModuleNotFoundError('PIL not found, you can install via `pip install pillow`.') from e


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
    pil_img = PILImage.open(src)
    img = from_pil(pil_img)

    return img


def from_zip(zip_src: Union[str, PosixPath], pattern: str = '*.JPEG') -> Image:
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
    path = str(Path(zip_src).resolve())
    with open(path, 'rb') as f:
        data = f.read()
    bio = io.BytesIO(data)
    zio = zipfile.ZipFile(bio)
    path_list = glob.fnmatch.filter([x.filename for x in zio.filelist], pattern)
    img_list = []
    for path in path_list:
        with zio.open(path) as f:
            pil_img = PILImage.open(f)
            img = from_pil(pil_img)
            img_list.append(img)

    return img_list


def from_pil(pil_img: PILImage.Image) -> Image:
    '''
    Convert a PIL.Image.Image into towhee.types.Image.

    Args:
        pil_img (`PIL.Image.Image`):
            A PIL image.

    Returns:
        (`towhee.types.Image`)
            The image wrapepd as towhee Image.
    '''
    img_bytes = pil_img.tobytes()
    img_width = pil_img.width
    img_height = pil_img.height
    img_channel = len(pil_img.split())
    img_mode = pil_img.mode
    img_array = np.array(pil_img)

    towhee_img = Image(img_bytes, img_width, img_height, img_channel, img_mode, img_array)

    return towhee_img


def to_pil(towhee_img: Image) -> PILImage.Image:
    """
    Convert a towhee.types.Image into PIL.Image.Image.

    Args:
        towhee_img (`towhee.types.Image`):
            A towhee image.

    Returns:
        (`PIL.Image.Image`)
            A PIL image.
    """
    mode = towhee_img.mode
    size = (towhee_img.width, towhee_img.height)
    data = towhee_img.image

    pil_img = PILImage.frombytes(mode, size, data)

    return pil_img
