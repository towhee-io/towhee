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

from towhee.utils.thirdparty.pyarrow import pa
from towhee.types.tensor_array import TensorArray
from towhee.hparam.hyperparameter import param_scope


class ColumnMixin:
    """
    Mixins to support column-based storage.
    """

    class ModeFlag(Flag):
        ROWBASEDFLAG = auto()
        COLBASEDFLAG = auto()
        HASCHUNKFLAG = auto()

    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_chunksize'):
            self._chunksize = parent._chunksize

    def set_chunksize(self, chunksize):
        """
        Set chunk size for arrow

        Examples:

        >>> import towhee
        >>> dc = towhee.dc['a'](range(100))
        >>> dc.set_chunksize(10)
        >>> dc = dc.runas_op['a', 'b'](func=lambda x: x+1)
        >>> dc.get_chunksize()
        10
        """
        self._chunksize = chunksize

    def get_chunksize(self):
        return self._chunksize

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

        >>> df.stream()._create_col_table()
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

        for entity in self._iterable:
            inner(entity)

        arrays = []
        for col in cols:
            try:
                arrays.append(pa.array(col))
            # pylint: disable=bare-except
            except:
                arrays.append(TensorArray.from_numpy(col))

        res = pa.Table.from_arrays(arrays, names=header)
        return res

    @classmethod
    def from_arrow_talbe(cls, **kws):
        arrays = []
        names = []
        for k, v in kws.items():
            arrays.append(v)
            names.append(k)
        return pa.Table.from_arrays(arrays, names=names)

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
        res = self._create_col_table()
        self._iterable = res
        self._mode = self.ModeFlag.COLBASEDFLAG

    def cmap(self, unary_op):
        # pylint: disable=protected-access
        res = self.__col_apply__(self._iterable, unary_op)
        arrays = [self._iterable[name] for name in self._iterable.column_names]
        names = self._iterable.column_names
        if isinstance(res, tuple):
            for x in res:
                arrays.append(TensorArray.from_numpy(x))
            for name in unary_op._index[1]:
                names.append(name)
        else:
            arrays.append(TensorArray.from_numpy(res))
            names.append(unary_op._index[1])
        table = pa.Table.from_arrays(arrays, names=names)
        return self._factory(table)

    def __col_apply__(self, data, unary_op):
        # pylint: disable=protected-access
        args = []
        if isinstance(unary_op._index[0], tuple):
            for col in unary_op._index[0]:
                args.append(args.append(data[col].chunks[0].as_numpy()))
        else:
            args.append(data[unary_op._index[0]].chunks[0].as_numpy())
        return unary_op.__vcall__(*args)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
