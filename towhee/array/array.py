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


class Array:
    """
    One-dimensional array of data
    """

    def __init__(self, name: str = None, data=None):
        """
        Args:
            name: the name of the Array
            data: supported data types are bool, int, float, str, bytes, PIL.Image,
                numpy.array, numpy.ndarray, and list of data mentioned above.
        """
        self.name = name
        self.dtype = None
        self.data = data
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, key: str):
        raise NotImplementedError

    @property
    def size(self) -> int:
        """
        Number of elements in the Array.
        """
        raise NotImplementedError

    @property
    def empty(self) -> bool:
        """
        Indicator whether Array is empty.
        True if Array has no elements.
        """

    def fill(self, value, size=None):
        """
        Fill the array with a scalar value.

        Args:
            value: all the Array elements will be assigned this value
            size: if not None, the Array will be resized.
        """
        raise NotImplementedError

    def to_bytes(self) -> bytes:
        raise NotImplementedError

    def from_bytes(self, value: bytes):
        raise NotImplementedError
