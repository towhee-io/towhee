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


class ListMixin:
    """
    Mixin to make dc an extension of python list.
    """
    def append(self, *args) -> 'DataCollection':
        """
        Append item to data collection.

        Args:
            item (Any): the item to append

        Returns:
            DataCollection: self

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([0, 1, 2])
        >>> dc.append(3).append(4)
        [0, 1, 2, 3, 4]
        """
        if hasattr(self._iterable, 'append'):
            self._iterable.append(*args)
            return self
        raise TypeError('append() is only supported for data collection created from list.')

    def clear(self, *args) -> 'DataCollection':
        """
        Clear a DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 2, 3])
        >>> dc.clear()
        []
        """
        if hasattr(self._iterable, 'clear'):
            self._iterable.clear(*args)
            return self
        raise TypeError('clear() is only supported for data collection created from list.')

    def copy(self, *args) -> 'DataCollection':
        """
        Copy a DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 2, 3])
        >>> dc_1 = dc.copy()
        >>> dc_1._iterable.append(4)
        >>> dc, dc_1
        ([1, 2, 3], [1, 2, 3, 4])
        """
        if hasattr(self._iterable, 'copy'):
            return self._factory(self._iterable.copy(*args))
        raise TypeError('copy() is only supported for data collection created from list.')

    def count(self, *args) -> int:
        """
        Count an element in  DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 2, 3])
        >>> dc.count(1)
        1
        """
        if hasattr(self._iterable, 'count'):
            return self._iterable.count(*args)
        raise TypeError('count() is only supported for data collection created from list.')

    def extend(self, *args) -> 'DataCollection':
        """
        Extend a DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 2, 3])
        >>> dc.extend([4, 5])
        [1, 2, 3, 4, 5]
        """
        if hasattr(self._iterable, 'extend'):
            self._iterable.extend(*args)
            return self
        raise TypeError('extend() is only supported for data collection created from list.')

    def insert(self, *args) -> 'DataCollection':
        """
        Insert data into a DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 2, 3])
        >>> dc.insert(0, 0)
        [0, 1, 2, 3]
        """
        if hasattr(self._iterable, 'insert'):
            self._iterable.insert(*args)
            return self
        raise TypeError('insert() is only supported for data collection created from list.')

    def pop(self, *args) -> 'DataCollection':
        """
        Extend a DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 2, 3])
        >>> dc.pop()
        [1, 2]
        """
        if hasattr(self._iterable, 'pop'):
            self._iterable.pop(*args)
            return self
        raise TypeError('pop() is only supported for data collection created from list.')

    def remove(self, *args) -> 'DataCollection':
        """
        Remove element from DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 2, 3])
        >>> dc.remove(1)
        [2, 3]
        """
        if hasattr(self._iterable, 'remove'):
            self._iterable.remove(*args)
            return self
        raise TypeError('remove() is only supported for data collection created from list.')

    def reverse(self, *args) -> 'DataCollection':
        """
        Reverse a DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 2, 3])
        >>> dc.reverse()
        [3, 2, 1]
        """
        if hasattr(self._iterable, 'reverse'):
            self._iterable.reverse(*args)
            return self
        raise TypeError('reverse() is only supported for data collection created from list.')

    def sort(self, *args) -> 'DataCollection':
        """
        Sort a DataCollection.

        Examples:

        >>> from towhee import DataCollection
        >>> dc = DataCollection([1, 4, 3])
        >>> dc.sort()
        [1, 3, 4]
        """
        if hasattr(self._iterable, 'sort'):
            self._iterable.sort(*args)
            return self
        raise TypeError('sort() is only supported for data collection created from list.')

    # def __len__(self):
    #     """
    #     Return the length of iterable.
    #     """
    #     if not self.is_stream():
    #         return self._iterable.__len__()
    #     else:
    #         raise TypeError('Streamed data collection does not support len.')
