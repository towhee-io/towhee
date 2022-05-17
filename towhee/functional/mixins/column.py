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
import pyarrow as pa


class ColumnMixin:
    """
    Mixins to support column-based storage.
    """
    def col_table(self):
        """
        Create a column-based table.

        Examples:

        >>> from towhee import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
        >>> df = DataFrame(e)
        >>> table = df.col_table()
        >>> table
        pyarrow.Table
        a: string
        b: int64
        ----
        a: [["abc","def","ghi"]]
        b: [[1,2,3]]
        """
        header = None
        cols = None

        # cols = [[getattr(entity, col) for entity in self._iterable] for col in header]
        def inner(entity):
            nonlocal cols, header
            header = [*entity.__dict__] if not header else header
            cols = [[] for _ in header] if not cols else cols
            for col, name in zip(cols, header):
                col.append(getattr(entity, name))

        any(map(inner, self._iterable))
        table = pa.Table.from_arrays(cols, names=header)

        return table

    def convert_entity(self, col_table):
        """
        Convert entities into column-based entities, with offset and col_table.

        Examples:

        >>> from towhee import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
        >>> df = DataFrame(e)
        >>> df
        [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]
        >>> col_table = df.col_table()
        >>> df.convert_entity(col_table)
        >>> df.to_list()
        [<Entity dict_keys(['offset', 'data'])>, <Entity dict_keys(['offset', 'data'])>, <Entity dict_keys(['offset', 'data'])>]
        """
        attrs = None

        def inner(off_ent):
            nonlocal attrs
            offset, entity = off_ent
            attrs = [*entity.__dict__] if not attrs else attrs
            for attr in attrs:
                delattr(entity, attr)
            setattr(entity, 'offset', offset)
            setattr(entity, 'data', col_table)

        any(map(inner, enumerate(self._iterable)))

    def to_column(self):
        """
        Convert the iterables to column-based.

        Examples:
        >>> from towhee import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
        >>> df = DataFrame(e)
        >>> df
        [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]
        >>> df.to_column()
        >>> df.to_list()
        [<Entity dict_keys(['offset', 'data'])>, <Entity dict_keys(['offset', 'data'])>, <Entity dict_keys(['offset', 'data'])>]
        """
        data = self.col_table()
        self.convert_entity(data)
        self._is_row = False

if __name__ == '__main__':
    import doctest
    doctest.testmod()
