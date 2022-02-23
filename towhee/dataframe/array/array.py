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


from typing import List, Union


class Array:
    """
    One-dimensional array of data

    Args:
        data (`numpy.ndarray`, `list`, `dict` or scalar):
            Contains data to be stored in `Array`.
        dtype (`numpy.dtype`, `str`, `PIL.Image` or `towhee.DType`, optional):
            Data type of the `Array`. If not specified, it will be inferred from `data`.
        name (`str`):
            Name of the `Array`.
        copy (`bool`, default False):
            Copy the `data` contents to the `Array`.
    """

    def __init__(
        self,
        data=None,
        # dtype=None,
        name: str = None,
        # copy=False
    ):

        self._name = name

        # For `data` is `list`
        if isinstance(data, list):
            pass
        # For `data` is `None`
        elif data is None:
            data = []
        # For `data` is scalar
        else:
            data = [data]

        self._data = data
        self._offset = 0

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            if key >= self._offset:
                return self._data[key - self._offset]
            elif key < 0:
                return self._data[len(self._data) + key]
            else:
                raise IndexError(f'element with index={key} has been released')
        elif isinstance(key, slice):
            if key.start >= self._offset:
                if key.stop is None:
                    stop = None
                else:
                    stop = key.stop - self._offset
                return self._data[key.start - self._offset:stop]
            else:
                raise IndexError(f'element with index={key.start} has been released' )
        else:
            raise IndexError('only integers are invalid indices')

    def __repr__(self):
        return self._data.__repr__()

    def __len__(self):
        """
        Number of elements in the `Array`.
        """
        return len(self._data)

    @property
    def size(self) -> int:
        """
        Total number of elements in the `Array` since creation.
        """
        return len(self._data) + self._offset

    @property
    def name(self) -> str:
        """
        Name of the Array'.
        """
        return self._name

    @property
    def data(self) -> List:
        """
        Data of the `Array`
        """
        return self._data

    def set_name(self, name):
        self._name = name

    def clear(self):
        self._data = []
        self._offset = 0

    def get_relative(self, key: int):
        return self._data[key + self._offset]

    def is_empty(self) -> bool:
        """
        Indicator whether `Array` is empty.

        Returns:
            (`bool`)
                Return `True` if `Array` has no elements.
        """
        return not self.size

    def put(self, item):
        """
        Append one item to the end of this `Array`.
        """
        self._data.append(item)

    def append(self, data):
        """
        Append a list-like items to the end of this `Array`.

        Args:
            data (`list` or `Array`):
                The data to be appended.
        """
        if isinstance(data, Array):
            self._data.extend(data.data)
        elif isinstance(data, List):
            self._data.extend(data)

   # TODO: Check out of bounds cases.
    def gc(self, offset):
        """
        Release the unreferenced lower part of the `Array` up to the index but not including.
        """

        if offset == -1:
            self.clear()
            return
        if offset > self.size:
            offset = self.size

        release_offset = offset - self._offset
        if release_offset > 0:
            del self._data[:release_offset]
            self._offset = offset

if __name__ == '__main__':
    arr = Array()
    arr.append([0, 1, 2, 3])
    arr.gc(4)
    arr.append([4])
    print(arr[4])
    print(arr)
