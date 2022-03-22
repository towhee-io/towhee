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

import numpy as np


class Image(np.ndarray):
    """
    This class represents an image object. The image data is a numpy.ndarray.

    Agrs:
        mode (`str`):
            The mode of the image(i.e. 'RGB', 'BGR', 'RGBA', 'HSV', etc.).
    """

    def __new__(cls, data: np.ndarray, mode: str = None):
        # Cast `np.ndarray` to be `Image`.
        # See https://numpy.org/doc/stable/user/basics.subclassing.html for details.
        obj = np.asarray(data).view(cls)
        obj._mode = mode
        return obj

    def __array_finalize__(self, obj):
        # `self` is a new object resulting from super().__new__(cls, ...), therefore it
        # only has attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        if obj is None:
            return
        self._mode = getattr(obj, '_mode', None)

    @property
    def mode(self):
        return self._mode
