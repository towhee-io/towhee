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

import threading
from typing import Iterable, List, Tuple, Any

from towhee.array import Array
import weakref


class DataFrame:
    """
    A `DataFrame` is a collection of immutable, potentially heterogeneous blogs of data.

    Args:
        name (`str`):
            Name of the dataframe; `DataFrame` names should be the same as its
            representation.
        data (`list[towhee.Array]` or `list[Tuple]` or `dict[str, towhee.Array]`):
            The data of the `DataFrame`. Internally, the data will be organized
            in a column-based manner.
    """

    def __init__(
        self,
        columns: List[Tuple[str, any]] = None,
        name: str = None,
        data=None,
    ):
        self._name = name
        self._len = 0
        self._sealed = False
        self._lock = threading.Lock()
        if columns != None:
            self._types = {x[0]: x[1] for x in columns}

        # For `data` is empty
        if not data:
            pass

        # For `data` is `list`
        elif isinstance(data, list):
            container_types = set(type(i) for i in data)
            if len(container_types) != 1:
                raise ValueError(
                    'can not construct Dataframe from a list of hybrid data containers. Try list[Tuple] or list[Array].')
            container_type = container_types.pop()

            # For `data` is `list[tuple]`
            if container_type is tuple:
                self._from_tuples(data, columns)
            # For `data` is `list[towhee.Array]`
            elif container_type is Array:
                self._from_arrays(data, columns)
            else:
                raise ValueError('can not construct DataFrame from list[%s]' % (container_type))

        # For `data` is `dict`
        elif isinstance(data, dict):
            self._from_dict(data)

        # Unrecognized data types
        else:
            raise ValueError('can not construct DataFrame from data type %s' % (type(data)))

    def __getitem__(self, key):
        # access a row
        if isinstance(key, int):
            return tuple(self._data_as_list[i][key] for i in range(len(self._data_as_list)))
        # access a column
        elif isinstance(key, str):
            return self._data_as_dict[key]
    
    def __str__(self):
        ret = ''
        formater = ''
        columns = []
        for x in range(len(self._data_as_list)):
            columns.append(self._data_as_list[x].name)
            formater += '{' + str(x) + ':30}'
        ret += formater.format(*columns) + '\n'

        for x in range(self._len):
            values = []
            for i in range(len(self._data_as_list)):
                values.append(str(self._data_as_list[i][x]))
            ret += formater.format(*values) + '\n'

        return ret

    def __len__(self):
        return self._len

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> List[Array]:
        return self._data_as_list

    def iter(self) -> Iterable[Tuple[Any, ...]]:
        """
        Iterate over DataFrame rows as tuples.
        """
        return DFIterator(self)

    def seal(self):
        with self._lock:
            self._sealed = True

    def is_sealed(self) -> bool:
        with self._lock:
            return self._sealed

    def _from_tuples(self, data, columns):

        # check tuple length
        tuple_lengths = set(len(i) for i in data)
        if len(tuple_lengths) == 1:
            tuple_length = tuple_lengths.pop()
        else:
            raise ValueError('can not construct DataFrame from unequal-length tuples')

        # check columns length
        if columns and len(columns) != tuple_length:
            raise ValueError('length of columns is not equal to the length of tuple')

        # create arrays
        if columns:
            self._data_as_list = [Array(name=columns[i][0]) for i in range(tuple_length)]
            self._data_as_dict = {columns[i][0]: self._data_as_list[i] for i in range(tuple_length)}
        else:
            self._data_as_list = [Array()] * tuple_length
            self._data_as_dict = None

        # tuples to arrays
        for row in data:
            for i, element in enumerate(row):
                self._data_as_list[i].put(element)

        self._len = len(data)

    def _from_arrays(self, data, columns):

        # check array length
        array_lengths = set(len(array) for array in data)
        if len(array_lengths) != 1:
            raise ValueError('arrays in data should have equal length')

        self._len = array_lengths.pop()

        # check columns length
        if columns and len(columns) != len(data):
            raise ValueError('length of columns is not equal to the number of arrays')

        self._data_as_list = data

        if columns:
            self._data_as_dict = {columns[i][0]: self._data_as_list[i] for i in range(len(data))}
        else:
            self._data_as_dict = None

    def _from_dict(self, data):

        # check dict values
        self._types = {}
        for key, value in data.items():
            self._types[key[0]] = key[1]
            if not isinstance(value, Array):
                raise ValueError('value type in data should be towhee.Array')

        # check arrays length
        array_lengths = set(len(array) for array in data.values())
        if len(array_lengths) != 1:
            raise ValueError('arrays in data should have equal length')

        self._len = array_lengths.pop()

        self._data_as_list = list(data.values())
        self._data_as_dict = data


class DFIterator:
    """
    A row-based `DataFrame` iterator.
    """

    def __init__(self, df: DataFrame):
        self._df = weakref.ref(df)
        self._offset = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns:
            (`Tuple[Any, ...]`)
                In the normal case, the iterator will return a `Tuple` at each call.
            (`None`)
                In the case that the `DataFrame` is not sealed and the new rows are
                not ready yet, the iterator will return `None`. The caller should
                determine whether to block the iteration or exit the loop.
        Raises:
            (`StopIteration`)
                The iteration end iff the `DataFrame` is sealed and the last row is
                reached.
        """

        if len(self._df) == self._offset:
            if self._df.is_sealed():
                # Already reach the last row
                raise StopIteration
            else:
                # No more ready rows
                return None
        else:
            row = self._df[self._offset]
            self._offset += 1
            return row

    def ack(self):
        """
        To notice the DataFrame that the iterated rows has been successfully processed.

        An acknowledgement (ack) will notice the `DataFrame`s that the rows already
        iterated over are no longer used, and can be deleted from the system.
        """
        pass
