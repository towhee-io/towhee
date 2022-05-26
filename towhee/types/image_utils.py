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

from towhee._types import Image


def to_image_color(img: Image, target_mode: str):
    """
    convert images from one color-space to another, like BGR ↔ Gray, BGR ↔ HSV, etc.
    """
    # pylint: disable=import-outside-toplevel
    from towhee.utils.cv2_utils import cv2
    if not hasattr(img, 'mode'):
        return img
    if img.mode == target_mode:
        return img

    flag_name = 'COLOR_' + img.mode.upper() + '2' + target_mode.upper()
    flag = getattr(cv2, flag_name, None)

    if flag is None:
        raise ValueError('Can not convert image from %s to %s.' % (img.mode, target_mode))

    return Image(cv2.cvtColor(img, flag), target_mode.upper())


def from_pil(pil_img):
    '''
    Convert a PIL.Image.Image into towhee.types.Image.

    Args:
        pil_img (`PIL.Image.Image`):
            A PIL image.

    Returns:
        (`towhee.types.Image`)
            The image wrapepd as towhee Image.
    '''
    # pylint: disable=import-outside-toplevel
    import numpy as np

    return Image(np.array(pil_img), pil_img.mode)


def to_pil(img: Image):
    """
    Convert a towhee.types.Image into PIL.Image.Image.

    Args:
        img (`towhee.types.Image`):
            A towhee image.

    Returns (`PIL.Image.Image`):
            A PIL image.
    """
    # pylint: disable=import-outside-toplevel
    from PIL import Image as PILImage

    return PILImage.fromarray(img, img.mode)
