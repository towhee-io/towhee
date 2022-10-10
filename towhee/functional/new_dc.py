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
from towhee.functional import Entity
from towhee.functional.mixins import DisplayMixin
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
        >>> from towhee.functional.new_dc import DataCollection
        >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
        >>> dq.put(('a', 'b1'))
        >>> DataCollection(dq)
        <DataCollection a: ColumnType.SCALAR, b: ColumnType.QUEUE>
    """
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        """
        Iterate the DataCollection in the form of Entity.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.functional.new_dc import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            >>> dq.put(('a', 'b2'))
            >>> dc = DataCollection(dq)
            >>> [i for i in dc]
            [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]
        """
        for _ in range(self._data.size):
            vals = self._data.get()
            keys = [k[0] for k in self._data.schema]
            yield Entity.from_dict(dict(zip(keys, vals)))

    def __add__(self, another: 'DataCollection') -> 'DataCollection':
        """
        Concat two DataCollections with same Schema.

        Note that this function will consume tha data in the second DataCollection.

        Args:
            another ('DataCollection'):
                Another DataCollection to concat.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.functional.new_dc import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq1 = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq2 = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq1.put(('a', 'b1'))
            >>> dq2.put(('a', 'b2'))
            >>> dc1 = DataCollection(dq1)
            >>> dc2 = DataCollection(dq2)
            >>> len(dc1)
            1
            >>> len(dc2)
            1
            >>> len(dc1 + dc2)
            2
        """
        for entity in another:
            self._data.put(list(entity.__dict__.values()))

        return self

    def __repr__(self) -> str:
        """
        String representation of the DataCollection.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.functional.new_dc import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dc = DataCollection(dq)
            >>> repr(dc)
            '<DataCollection a: ColumnType.SCALAR, b: ColumnType.QUEUE>'
        """
        names = self._data._schema.col_names
        types = self._data._schema.col_types
        content = ', '.join([i + ': ' + str(j) for i, j in zip(names, types)])
        return f'<{self.__class__.__name__} {content}>'

    def __len__(self):
        """
        Return the number of entities in the DataCollection.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.functional.new_dc import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dc = DataCollection(dq)
            >>> len(dc)
            0
        """
        return self._data.size

    def to_list(self) -> list:
        """
        Convert DataCollection to list.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.functional.new_dc import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            >>> dc = DataCollection(dq)
            >>> dc.to_list()
            [<Entity dict_keys(['a', 'b'])>]
        """
        return list(self)


    def copy(self):
        """
        Copy a DataCollection.

        Examples:
            >>> from towhee.runtime.data_queue import DataQueue, ColumnType
            >>> from towhee.functional.new_dc import DataCollection
            >>> dq = DataQueue([('a', ColumnType.SCALAR), ('b', ColumnType.QUEUE)])
            >>> dq.put(('a', 'b1'))
            >>> dc = DataCollection(dq)
            >>> dc_copy = dc.copy()
            >>> id(dc) == id(dc_copy)
            False
        """
        return DataCollection(self._data.copy())
