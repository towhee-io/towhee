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

import copy
import numpy as np


class Image:
    """
    The unified form of images defined in Towhee.

    Agrs:
        image (`bytes`):
            The bytes of the image.
        width (`int`):
            The width of the image.
        height (`int`):
            The height of the image.
        channel (`int`):
            The channel of the image.
        mode (`str`):
            The mode of the image(i.e. 'RGB', 'RGBA', 'HSV', etc.).
        array (`np.ndarray`):
            The image in the form of ndarray.
    """

    def __init__(self, image: bytes, width: int, height: int, channel: int, mode: str, array: np.ndarray = None, key_frame: bool = False):
        self._image = image
        self._width = width
        self._height = height
        self._channel = channel
        self._mode = mode
        self._array = array
        self._key_frame = key_frame

    @property
    def image(self) -> bytes:
        return self._image

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def channel(self) -> int:
        return self._channel

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def key_frame(self):
        return self._key_frame

    @property
    def array(self) -> np.ndarray:
        if not isinstance(self._array, np.ndarray):
            raise AttributeError('The array of image is not given, please call `Image.to_ndarray()` function to get the ndarray.')
        else:
            return copy.deepcopy(self._array)

    def to_ndarray(self) -> np.ndarray:
        """
        Load the np.ndarray form of the image.
        """
        if not isinstance(self._array, np.ndarray):
            shape = (self._height, self._width, self._channel)
            self._array = np.ndarray(shape, np.uint8, self._image)
            # self._array = np.frombuffer(self._image, dtype=np.uint8).reshape(self._height, self._width, self._channel)

        return copy.deepcopy(self._array)
