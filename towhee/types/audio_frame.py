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


class AudioFrame(np.ndarray):
    """
    This class represents an audio frame. The data is a numpy.ndarray.

    Args:
        sample_rate (`int`):
            The audio sample rate in Hz.
    """

    def __new__(cls, data: np.ndarray, sample_rate: int = None):
        # Cast `np.ndarray` to be `AudioFrame`.
        # See https://numpy.org/doc/stable/user/basics.subclassing.html for details.
        obj = np.asarray(data).view(cls)
        obj._sample_rate = sample_rate
        return obj

    def __array_finalize__(self, obj):
        # `self` is a new object resulting from super().__new__(cls, ...), therefore it
        # only has attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        if obj is None:
            return
        self._sample_rate = getattr(obj, '_sample_rate', None)

    @property
    def sample_rate(self):
        return self._sample_rate
