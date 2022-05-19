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
from enum import Flag, auto

import pyarrow as pa


class ColumnMixin:
    """
    Mixins to support column-based storage.
    """
    class ModeFlag(Flag):
        ROWBASEDFLAG = auto()
        COLBASEDFLAG = auto()

    def _create_col_table(self):
        """
        Create a column-based table.

        Examples:

        >>> from towhee import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
        >>> df = DataFrame(e)
        >>> table = df._create_col_table()
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

        # TODO(KY): map?
        any(map(inner, self._iterable))
        table = pa.Table.from_arrays(cols, names=header)

        return table

    def to_column(self):
        """
        Convert the iterables to column-based table.

        Examples:

        >>> from towhee import Entity, DataFrame
        >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
        >>> df = DataFrame(e)
        >>> df
        [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]
        >>> df.to_column()
        >>> df
        pyarrow.Table
        a: string
        b: int64
        ----
        a: [["abc","def","ghi"]]
        b: [[1,2,3]]
        """
        self._iterable = self._create_col_table()
        self._mode = self.ModeFlag.COLBASEDFLAG


if __name__ == '__main__':
    import doctest
    doctest.testmod()
