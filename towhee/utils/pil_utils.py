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

from pathlib import PosixPath
from typing import Union
import numpy as np

from towhee.utils.log import engine_log
from towhee.types import Image

try:
    # pylint: disable=unused-import,ungrouped-imports
    from PIL import Image as PILImage
except ModuleNotFoundError as moduleNotFound:
    try:
        from towhee.utils.dependency_control import prompt_install
        prompt_install('pillow')
        # pylint: disable=unused-import,ungrouped-imports
        from PIL import Image as PILImage
    except:
        engine_log.error('PIL not found, you can install via `pip install pillow`.')
        raise ModuleNotFoundError('PIL not found, you can install via `pip install pillow`.') from moduleNotFound


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
    # img_bytes = pil_img.tobytes()
    # img_width = pil_img.width
    # img_height = pil_img.height
    # img_channel = len(pil_img.split())
    mode = pil_img.mode
    array = np.array(pil_img)

    t_img = Image(array, mode)

    return t_img


def to_pil(t_img: Image) -> PILImage.Image:
    """
    Convert a towhee.types.Image into PIL.Image.Image.

    Args:
        t_img (`towhee.types.Image`):
            A towhee image.

    Returns:
        (`PIL.Image.Image`)
            A PIL image.
    """
    mode = t_img.mode

    pil_img = PILImage.fromarray(t_img, mode)

    return pil_img
