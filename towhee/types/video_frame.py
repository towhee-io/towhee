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


class VideoFrame(np.ndarray):
    """
    This class represents an image object. The image data is a numpy.ndarray.

    Agrs:
        mode (`str`):
            The mode of the image(i.e. 'RGB', 'BGR', 'RGBA', 'HSV', etc.).
    """

    def __new__(cls, data: np.ndarray, mode: str = None, timestamp: int = None, key_frame: int = 0):
        # Cast `np.ndarray` to be `Image`.
        # See https://numpy.org/doc/stable/user/basics.subclassing.html for details.
        obj = np.asarray(data).view(cls)
        obj._mode = mode
        obj._timestamp = timestamp
        obj._key_frame = key_frame
        return obj

    def __array_finalize__(self, obj):
        # `self` is a new object resulting from super().__new__(cls, ...), therefore it
        # only has attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        if obj is None:
            return
        self._mode = getattr(obj, '_mode', None)
        self._timestamp = getattr(obj, '_timestamp', None)
        self._key_frame = getattr(obj, '_key_frame', 0)

    def __str__(self):
        return f'VideoFrame shape: {self.shape}, mode: {self.mode}, timestamp: {self.timestamp}, key_frame: {self.key_frame}'

    def __reduce__(self):
        # Get numpy pickle
        pickled_state = super(VideoFrame, self).__reduce__() #pylint: disable=super-with-arguments
        # Attach the attributes to the numpy pickle
        new_state = pickled_state[2] + (self.__dict__,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # Set attributes from the pickle
        self.__dict__.update(state[-1])
        # Call the parent's __setstate__ with the other tuple elements.
        super(VideoFrame, self).__setstate__(state[0:-1]) #pylint: disable=super-with-arguments

    @property
    def mode(self):
        return self._mode

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def key_frame(self):
        return self._key_frame
