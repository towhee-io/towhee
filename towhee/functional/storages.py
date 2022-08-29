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
from .entity import EntityView
# pylint: disable=import-outside-toplevel


class WritableTable:
    """A wrapper that make arrow table writable.

        Args:
            table (pyarrow.Table): The pyarrow table to write.
        """
    def __init__(self, table):
        self._table = table
        self._buffer = {}

    def write_many(self, names, arrays):
        """Write many arrays to table.

        Args:
            names (str): The name of column.
            arrays (any): The values of the column.

        Returns:
            pyarrow.Table: The new table.
        """
        self.prepare()
        if isinstance(arrays, tuple):
            self._buffer = dict(zip(names, arrays))
        else:
            self._buffer[names] = arrays
        return self.seal()

    def write(self, name, offset, value):
        """Write a value to a column.

        Args:
            name (str): Column name.
            offset (int): The offset to write the value at.
            value (any): The value to write.
        """
        if name not in self._buffer:
            self._buffer[name] = []
        while len(self._buffer[name]) < offset:
            self._buffer[name].append(None)
        self._buffer[name].append(value)

    def prepare(self):
        """Return the WriteableTable.

        Returns:
            WriteableTable: The writeable table.
        """
        if not hasattr(self, '_sealed'):
            self._sealed = WritableTable(None)
        return self._sealed

    def seal(self):
        """Writes the values from buffer to the table and seals the table.

        Returns:
            WriteableTable: The updated and sealed pyarrow.Table.
        """
        # pylint: disable=protected-access
        from towhee.utils.thirdparty.pyarrow_utils import pa
        from towhee.types.tensor_array import TensorArray

        names = list(self._buffer)
        arrays = []
        for name in names:
            try:
                arrays.append(pa.array(self._buffer[name]))
            # pylint: disable=bare-except
            except:
                arrays.append(TensorArray.from_numpy(self._buffer[name]))
        new_table = self._table
        for name, arr in zip(names, arrays):
            new_table = new_table.append_column(name, arr)
        self._sealed._table = new_table
        return self._sealed

    def __iter__(self):
        """Iterator.

        Yields:
            EntityView: The entity for each row.
        """
        for i in range(self._table.shape[0]):
            yield EntityView(i, self)

    def __len__(self):
        return self._table.num_rows

    def __getattr__(self, name):
        if name in self._table.column_names:
            return self._table.__getitem__(name)
        else:
            return getattr(self._table, name)

    def __getitem__(self, index):
        return self._table.__getitem__(index)

    def __repr__(self) -> str:
        return repr(self._table)


class ChunkedTable:
    """A chunked pyarrow table.

    Args:
        chunks:
            The list or queue of chunks.
        chunksize (`int`):
            The size of the chunk.
        stream (`bool`):
            If the data is streamed.
    """
    def __init__(self, chunks=None, chunksize=128, stream=False) -> None:
        self._chunksize = chunksize
        self._is_stream = stream
        if chunks is not None:
            self._chunks = chunks
        else:
            self._chunks = [] if not stream else None

    @property
    def is_stream(self):
        return self._is_stream

    @property
    def chunksize(self):
        return self._chunksize

    def chunks(self):
        return self._chunks

    def _create_table(self, chunk, head):
        from towhee.utils.thirdparty.pyarrow_utils import pa
        from towhee.types.tensor_array import TensorArray

        # head = []
        cols = None
        for entity in chunk:
            cols = [[] for _ in head] if cols is None else cols
            for col, name in zip(cols, head):
                col.append(getattr(entity, name))
        arrays = []
        for col in cols:
            try:
                arrays.append(pa.array(col))
            # pylint: disable=bare-except
            except:
                arrays.append(TensorArray.from_numpy(col))

        res = pa.Table.from_arrays(arrays, names=head)

        return res

    def _pack_unstream_chunk(self, data):
        head = []
        res = []
        chunk = []
        # if not self._column_names:
        #     self._column_names = [*data[0].__dict__]
        for element in data:
            if not head:
                head = [*element.__dict__]
            chunk.append(element)
            if len(chunk) >= self._chunksize:
                res.append(WritableTable(self._create_table(chunk, head)))
                chunk = []

        if len(chunk) != 0:
            res.append(WritableTable(self._create_table(chunk, head)))

        return res

    def _pack_stream_chunk(self, data):
        chunk = []
        head = []
        for element in data:
            if not head:
                head = [*element.__dict__]
            chunk.append(element)
            if len(chunk) >= self._chunksize:
                yield WritableTable(self._create_table(chunk, head))
                chunk = []

        if len(chunk) != 0:
            yield WritableTable(self._create_table(chunk, head))


    def feed(self, data):
        if not self._is_stream:
            self._chunks = self._pack_unstream_chunk(data)
        else:
            self._chunks = self._pack_stream_chunk(data)

    def __iter__(self):
        def inner():
            for chunk in self._chunks:
                # yield chunk
                for i in range(len(chunk)):
                    yield EntityView(i, chunk)

        return inner()
