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


from typing import List
from towhee.dataframe.array._array_ref import _ArrayRef


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

        self.name = name

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
        self._ref = _ArrayRef()
        self._offset = 0

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, key: int):
        if isinstance(key, int):
            if key >= self._offset:
                return self._data[key - self._offset]
            else:
                raise IndexError('element with index=%d has been released' % (key))
        else:
            raise IndexError('only integers are invalid indices')

    def __repr__(self):
        return self._data.__repr__()

    def __len__(self):
        """
        Number of elements in the `Array`.
        """
        return len(self._data) + self._offset

    @property
    def size(self) -> int:
        """
        Number of elements in the `Array`.
        """
        return len(self._data) + self._offset

    @property
    def physical_size(self) -> int:
        """
        Number of elements still existed in the `Array`
        """
        return len(self._data)

    @property
    def data(self) -> List:
        """
        Data of the `Array`
        """
        return self._data

    def is_empty(self) -> bool:
        """
        Indicator whether `Array` is empty.

        Returns:
            (`bool`)
                Return `True` if `Array` has no elements.
        """
        return not self.size

    def add_reader(self) -> int:
        """
        Add a read reference to `Array`
        """
        return self._ref.add_reader()

    def remove_reader(self, ref_id: int):
        """
        Remove a read reference from `Array`

        Args:
            ref_id (`int`):
                The reference ID
        """
        self._ref.remove(ref_id)

    def update_reader_offset(self, ref_id: int, offset: int):
        """
        Update a reference offset

        Args:
            ref_id (`int`):
                The reference ID
            offset (`int`):
                The new reference offset
        """
        self._ref.update_reader_offset(ref_id, offset)

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

    def gc(self):
        """
        Release the unreferenced upper part of the `Array`, if any
        """
        min_ref_offset = self._ref.min_reader_offsets
        release_offset = min_ref_offset - self._offset
        if release_offset > 0:
            del self._data[:release_offset]
            self._offset = min_ref_offset
