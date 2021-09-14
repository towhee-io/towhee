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
    """One-dimensional array of data

    Args:
        data: (`numpy.ndarray`, `list`, `dict` or scalar)
            Contains data to be stored in `Array`.
        dtype: (`numpy.dtype`, `str`, `PIL.Image` or `towhee.DType`, optional)
            Data type of the `Array`. If not specified, it will be inferred from `data`.
        name: (`str`)
            Name of the `Array`.
        copy: (`bool`, default False)
            Copy the `data` contents to the `Array`.
    """
    def __init__(
        self,
        data=None,
        #dtype=None,
        name: str = None,
        #copy=False
    ):

        self.name = name
        # TODO(GuoRentong): provide towhee.DType abstraction and support
        # Array.dtypes
        self.dtype = None

        # For `data` is `list`
        if isinstance(data, list):
            pass
        # For `data` is `None`
        elif data is None:
            # TODO(GuoRentong): remember to handle dtype properly
            data = []
        # For `data` is scalar
        else:
            data = [data]

        self._data = data

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, key: int):
        if isinstance(key, int):
            return self._data[key]
        else:
            raise IndexError("only integers are invalid indices")

    def __repr__(self):
        return self._data.__repr__()

    @property
    def size(self) -> int:
        """ Number of elements in the `Array`.
        """
        return len(self._data)

    def is_empty(self) -> bool:
        """
        Indicator whether Array is empty.
        True if Array has no elements.
        """
        return not self.size

    def put(self, item):
        """Append one item to the end of this `Array`.
        """
        self._data.append(item)

    def append(self, data):
        """Append a list-like items to the end of this `Array`.
        Args:
            data: (`list` or `Array`)
                The data to be appended.
        """
        if isinstance(data, Array):
            data = data._data
        self._data.extend(data)
