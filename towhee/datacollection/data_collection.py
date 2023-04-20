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
import copy
from typing import Any

from towhee.datacollection.entity import Entity
from towhee.runtime.data_queue import ColumnType
from towhee.datacollection.mixins.display import DisplayMixin


# pylint: disable=protected-access
class DataCollection(DisplayMixin):
    """
    A pythonic computation and processing framework.

    DataCollection is a pythonic computation and processing framework for unstructured
    data in machine learning and data science. It allows a data scientist or researcher
    to assemble data processing pipelines and do their model work (embedding,
    transforming, or classification) with a method-chaining style API.

    Args:
        data ('towhee.runtime.DataQueue'):
            The data to be stored in DataColletion in the form of DataQueue.

    Examples:
        >>> from towhee.runtime.data_queue import DataQueue, ColumnType
        >>> from towhee.datacollection.data_collection import DataCollection
        >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
        >>> dq.put(('a', 'b1'))
        True
        >>> DataCollection(dq)
        <DataCollection Schema[a: ColumnType.SCALAR, b: ColumnType.QUEUE] SIZE 1>
    """
    def __init__(self, data):
        if isinstance(data, dict):
            self._schema = data['schema']
            self._type_schema = [ColumnType[type] for type in data['type_schema']]
            self._iterable = [Entity.from_dict(dict(zip(data['schema'], entity))) for entity in data['iterable']]
        else:
            self._schema = data.schema
            self._type_schema = data.type_schema
            self._iterable = [Entity.from_dict(dict(zip(self._schema, data.get()))) for _ in range(data.size)]

    def __iter__(self):
        """
        Iterate the DataCollection in the form of Entity.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.datacollection.data_collection import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            True
            >>> dq.put(('a', 'b2'))
            True
            >>> dc = DataCollection(dq)
            >>> [i for i in dc]
            [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]
        """
        return iter(self._iterable)

    def __getitem__(self, index: int):
        """
        Get the item with given index.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.datacollection.data_collection import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            True
            >>> dc = DataCollection(dq)
            >>> dc[0]
            <Entity dict_keys(['a', 'b'])>
        """
        return self._iterable[index]

    def __setitem__(self, index: int, value: Any):
        """
        Set the item to given value.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.datacollection.data_collection import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            True
            >>> dc = DataCollection(dq)
            >>> dc[0] = 'a'
            >>> dc[0]
            'a'
        """
        self._iterable[index] = value

    def __repr__(self) -> str:
        """
        String representation of the DataCollection.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.datacollection.data_collection import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dc = DataCollection(dq)
            >>> repr(dc)
            '<DataCollection Schema[a: ColumnType.SCALAR, b: ColumnType.QUEUE] SIZE 0>'
        """
        names = self._schema
        types = self._type_schema
        content = ', '.join([i + ': ' + str(j) for i, j in zip(names, types)])
        return f'<{self.__class__.__name__} Schema[{content}] SIZE {len(self)}>'

    def __len__(self):
        """
        Return the number of entities in the DataCollection.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.datacollection.data_collection import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dc = DataCollection(dq)
            >>> len(dc)
            0
        """
        return len(self._iterable)

    def __add__(self, another: 'DataCollection') -> 'DataCollection':
        """
        Concat two DataCollections with same Schema.

        Note that this function will consume tha data in the second DataCollection.

        Args:
            another ('DataCollection'):
                Another DataCollection to concat.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.datacollection.data_collection import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq1 = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq2 = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq1.put(('a', 'b1'))
            True
            >>> dq2.put(('a', 'b2'))
            True
            >>> dc1 = DataCollection(dq1)
            >>> dc2 = DataCollection(dq2)
            >>> len(dc1)
            1
            >>> len(dc2)
            1
            >>> len(dc1 + dc2)
            2
        """
        new = copy.deepcopy(self)
        new._iterable = self._iterable + another._iterable

        return new

    def to_list(self) -> list:
        """
        Convert DataCollection to list.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.datacollection.data_collection import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            True
            >>> dc = DataCollection(dq)
            >>> dc.to_list()
            [<Entity dict_keys(['a', 'b'])>]
        """
        return list(self)

    def copy(self, deep: bool = False):
        """
        Copy a DataCollection.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.datacollection.data_collection import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            True
            >>> dc = DataCollection(dq)
            >>> dc_copy = dc.copy()
            >>> dc_dcopy = dc.copy(True)
            >>> id(dc) == id(dc_copy)
            False
            >>> id(dc[0]) == id(dc_copy[0])
            True
            >>> id(dc) == id(dc_dcopy)
            False
            >>> id(dc[0]) == id(dc_dcopy[0])
            False
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def to_dict(self):
        ret = {}
        ret['schema'] = self._schema
        ret['type_schema'] = [type.name for type in self._type_schema]
        ret['iterable'] = [[getattr(entity, col) for col in self._schema] for entity in self._iterable]

        return ret

    @staticmethod
    def from_dict(data):
        return DataCollection(data)
