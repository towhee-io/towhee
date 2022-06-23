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
from towhee.utils.thirdparty.pyarrow import pa
from towhee.types.tensor_array import TensorArray


class WritableTable:
    """
    A wrapper that make arrow table writable.
    """

    def __init__(self, table):
        self._table = table
        self._buffer = {}

    def write_many(self, names, arrays):
        self.prepare()
        if isinstance(arrays, tuple):
            self._buffer = dict(zip(names, arrays))
        else:
            self._buffer[names] = arrays
        return self.seal()

    def write(self, name, offset, value):
        if name not in self._buffer:
            self._buffer[name] = []
        while len(self._buffer[name]) < offset:
            self._buffer[name].append(None)
        self._buffer[name].append(value)

    def prepare(self):
        if not hasattr(self, '_sealed'):
            self._sealed = WritableTable(None)
        return self._sealed

    def seal(self):
        # pylint: disable=protected-access
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
        for i in range(self._table.shape[0]):
            yield EntityView(i, self)

    def __len__(self):
        return self._table.num_rows

    def __getattr__(self, name):
        return getattr(self._table, name)

    def __getitem__(self, index):
        return self._table.__getitem__(index)

    def __repr__(self) -> str:
        return repr(self._table)


class ChunkedTable:
    """
    Chunked arrow table
    """

    def __init__(self, chunks=None, chunksize=128, stream=False) -> None:

        self._chunksize = chunksize
        self._buffer = []
        if chunks is not None:
            self._chunks = chunks
        else:
            self._chunks = [] if stream is False else None

    def feed(self, element, eos=False):
        if not eos:
            self._buffer.append(element)

        if len(self._buffer) >= self._chunksize or eos is True:
            if len(self._buffer) == 0: return
            header = None
            cols = None
            for entity in self._buffer:
                header = [*entity.__dict__] if header is None else header
                cols = [[] for _ in header] if cols is None else cols
                for col, name in zip(cols, header):
                    col.append(getattr(entity, name))
            arrays = []
            for col in cols:
                try:
                    arrays.append(pa.array(col))
                # pylint: disable=bare-except
                except:
                    arrays.append(TensorArray.from_numpy(col))

            res = pa.Table.from_arrays(arrays, names=header)
            self._chunks.append(WritableTable(res))
            self._buffer = []
        return


    def chunks(self):
        return self._chunks

    def __iter__(self):

        def inner():
            for chunk in self._chunks:
                # chunk = _WritableTable(chunk)
                for i in range(len(chunk)):
                    yield EntityView(i, chunk)

        return inner()
