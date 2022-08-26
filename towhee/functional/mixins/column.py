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

from towhee.hparam.hyperparameter import param_scope

from towhee.functional.storages import ChunkedTable, WritableTable


# pylint: disable=import-outside-toplevel
# pylint: disable=bare-except
class ColumnMixin:
    """Mixins to support column-based storage.
    """

    class ModeFlag(Flag):
        ROWBASEDFLAG = auto()
        COLBASEDFLAG = auto()
        CHUNKBASEDFLAG = auto()

    def __init__(self) -> None:
        super().__init__()
        with param_scope() as hp:
            parent = hp().data_collection.parent(None)
        if parent is not None and hasattr(parent, '_chunksize'):
            self._chunksize = parent._chunksize

    def set_chunksize(self, chunksize):
        """Set chunk size for arrow

        Args:
            chuksize (int): How many rows per chunk.

        Return:
            DataCollection: New DataCollection converted to Table.

        Examples:

            >>> import towhee
            >>> dc_1 = towhee.dc['a'](range(20))
            >>> dc_1 = dc_1.set_chunksize(10)
            >>> dc_2 = dc_1.runas_op['a', 'b'](func=lambda x: x+1)
            >>> dc_1.get_chunksize(), dc_2.get_chunksize()
            (10, 10)
            >>> dc_2._iterable.chunks()
            [pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[0,1,2,3,4,5,6,7,8,9]]
            b: [[1,2,3,4,5,6,7,8,9,10]], pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[10,11,12,13,14,15,16,17,18,19]]
            b: [[11,12,13,14,15,16,17,18,19,20]]]

            >>> dc_3 = towhee.dc['a'](range(20)).stream()
            >>> dc_3 = dc_3.set_chunksize(10)
            >>> dc_4 = dc_3.runas_op['a', 'b'](func=lambda x: x+1)
            >>> dc_4._iterable.chunks()
            [pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[0,1,2,3,4,5,6,7,8,9]]
            b: [[1,2,3,4,5,6,7,8,9,10]], pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[10,11,12,13,14,15,16,17,18,19]]
            b: [[11,12,13,14,15,16,17,18,19,20]]]
        """

        self._chunksize = chunksize
        chunked_table = ChunkedTable(chunksize=chunksize, stream=self.is_stream)
        chunked_table.feed(self._iterable)
        return self._factory(chunked_table, parent_stream=False, mode=self.ModeFlag.CHUNKBASEDFLAG)

    def get_chunksize(self):
        return self._chunksize

    def _create_col_table(self):
        """Create a column-based table.

        Returns:
            pyarrow.Table: The columnar table.

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
        from towhee.utils.thirdparty.pyarrow_utils import pa
        from towhee.types.tensor_array import TensorArray

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

        return pa.Table.from_arrays(arrays, names=header)

    @classmethod
    def create_arrow_table(cls, **kws):
        """Convert kwargs to Table.

        Returns:
            pyarrow.Table: The Table from the kwargs.
        """
        from towhee.utils.thirdparty.pyarrow_utils import pa

        arrays = []
        names = []
        for k, v in kws.items():
            arrays.append(v)
            names.append(k)
        return pa.Table.from_arrays(arrays, names=names)

    def to_column(self):
        """Convert the DataCollection to column-based table DataCollection.

        Returns:
            DataCollection: The current DC converted to Table DC.

        Examples:

            >>> from towhee import Entity, DataFrame
            >>> e = [Entity(a=a, b=b) for a,b in zip(['abc', 'def', 'ghi'], [1,2,3])]
            >>> df = DataFrame(e)
            >>> df
            [<Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>, <Entity dict_keys(['a', 'b'])>]
            >>> df.to_column()
            pyarrow.Table
            a: string
            b: int64
            ----
            a: [["abc","def","ghi"]]
            b: [[1,2,3]]
        """

        # pylint: disable=protected-access
        df = self.to_df()
        res = df._create_col_table()
        df._iterable = WritableTable(res)
        df._mode = self.ModeFlag.COLBASEDFLAG
        return df

    def cmap(self, unary_op):
        """Chunked map.

        Args:
            unary_op (callable): The operation to map.

        Returns:
            DataCollection: A new DataCollection after mapping.

        Examples:

            >>> import towhee
            >>> dc = towhee.dc['a'](range(10))
            >>> dc = dc.to_column()
            >>> dc = dc.runas_op['a', 'b'](func=lambda x: x+1)
            >>> dc.show(limit=5, tablefmt='plain')
              a    b
              0    1
              1    2
              2    3
              3    4
              4    5
            >>> dc._iterable
            pyarrow.Table
            a: int64
            b: int64
            ----
            a: [[0,1,2,3,4,5,6,7,8,9]]
            b: [[1,2,3,4,5,6,7,8,9,10]]
            >>> len(dc._iterable)
            10
        """
        # pylint: disable=protected-access
        if self.get_executor() is None:
            if isinstance(self._iterable, ChunkedTable):
                if not self.is_stream:
                    tables = [WritableTable(self.__table_apply__(chunk, unary_op)) for chunk in self._iterable.chunks()]
                else:
                    tables = (WritableTable(self.__table_apply__(chunk, unary_op)) for chunk in self._iterable.chunks())
                return self._factory(ChunkedTable(chunks = tables))
            return self._factory(self.__table_apply__(self._iterable, unary_op))
        else:
            return self.pmap(unary_op)

    def __table_apply__(self, table, unary_op):
        # pylint: disable=protected-access
        return table.write_many(unary_op._index[1],
                                self.__col_apply__(table, unary_op))

    def __col_apply__(self, cols, unary_op):
        """Columnar apply.

        Args:
            cols (list): List of columns to apply op to.
            unary_op (callable): The operation

        Returns:
            DataCollection: The datacollection with callable applied.
        """
        # pylint: disable=protected-access
        from towhee.utils.thirdparty.pyarrow_utils import pa
        from towhee.types.tensor_array import TensorArray

        import numpy as np
        args = []
        # Multi inputs.
        if isinstance(unary_op._index[0], tuple):
            for name in unary_op._index[0]:
                try:
                    data = cols[name].combine_chunks()
                except:
                    data = cols[name].chunk(0)

                buffer = data.buffers()[-1]
                dtype = data.type
                if isinstance(data, TensorArray):
                    dtype = dtype.storage_type.value_type
                elif hasattr(data.type, 'value_type'):
                    while hasattr(dtype, 'value_type'):
                        dtype = dtype.value_type
                dtype = dtype.to_pandas_dtype()
                shape = [-1, *data.type.shape] if isinstance(data, TensorArray)\
                    else [len(data), -1] if isinstance(data, pa.lib.ListArray)\
                    else [len(data)]
                args.append(
                    np.frombuffer(buffer=buffer, dtype=dtype).reshape(shape))
                # args.append(data.to_numpy(zero_copy_only=False).reshape(shape))

        # Single input.
        else:
            try:
                data = cols[unary_op._index[0]].combine_chunks()
            except:
                data = cols[unary_op._index[0]].chunk(0)

            buffer = data.buffers()[-1]
            dtype = data.type
            if isinstance(data, TensorArray):
                dtype = dtype.storage_type.value_type
            elif hasattr(data.type, 'value_type'):
                while hasattr(dtype, 'value_type'):
                    dtype = dtype.value_type
            dtype = dtype.to_pandas_dtype()
            shape = [-1, *data.type.shape] if isinstance(data, TensorArray)\
                else [len(data), -1] if isinstance(data, pa.lib.ListArray)\
                else [len(data)]
            args.append(
                np.frombuffer(buffer=buffer, dtype=dtype).reshape(shape))
            # args.append(data.to_numpy(zero_copy_only=False).reshape(shape))

        return unary_op.__vcall__(*args)
