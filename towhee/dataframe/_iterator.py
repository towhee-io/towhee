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


from towhee.dataframe.dataframe import DataFrame
from typing import Callable


class _BaseIterator:
    """Base iterator implementation. All iterators should be subclasses of
    `_BaseIterator`.

    Args:
        df: The dataframe to iterate over.
    """

    def __init__(self, df: DataFrame):
        self._df = df

    def __iter__(self):
        return self

    def __next__(self):
        # The base iterator is purposely defined to have exatly 0 elements.
        raise StopIteration


class MapIterator:
    """Iterator implementation that traverses the dataframe line-by-line.

    Args:
        df: The dataframe to iterate over.
    """

    def __init__(self, df: DataFrame):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class BatchIterator:
    """Iterator implementation that traverses the dataframe multiple rows at a time.

    Args:
        df:
            The dataframe to iterate over.
        size:
            batch size. If size is None, then the whole variable set will be batched
            together.
    """

    def __init__(self, df: DataFrame, size: int = None):
        super().__init__(df)
        self._size = size

    def __next__(self):
        raise NotImplementedError


class GroupIterator:
    """Iterator implementation that traverses the dataframe based on a custom group-by
    function.

    Args:
        df:
            The dataframe to iterate over.
        func:
            The function used to return grouped data within the dataframe.
    """

    def __init__(self, df: DataFrame, func: Callable[[], None]):
        super().__init__(df)
        self._func = func

    def __next__(self):
        raise NotImplementedError


class RepeatIterator:
    """Iterator implementation that repeatedly accesses the first line of the dataframe.

        Args:
            df:
                The dataframe to iterate over.
            n:
                the repeat times. If `None`, the iterator will loop continuously.
    """

    def __init__(self, df: DataFrame, n: int = None):
        # self._n = n
        # self._i = 0
        raise NotImplementedError

    def __next__(self):
        if self._i >= self._n:
            raise StopIteration
        return self._data[0]
