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

from abc import ABC, abstractmethod
import threading
from collections import namedtuple
from typing import Dict, Tuple, Union, List


class ReaderBase(ABC):
    """
    The reader base class.

    The read() could be blocking or non-blocking function, if it's a blocking function,
    the runner may be blocked. When need to stop the graph, we call close to interrupting it.
    """

    @abstractmethod
    def read(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class DataFrameReader(ReaderBase):
    """
    Read data from input dataframes, unpack and combine data.
    One op_ctx has one dataframe reader.
    """

    def __init__(self, input_readers: List[iter], input_order: List):
        self._iters = input_readers
        self._iters_count = len(self._iters)
        self._input_order = input_order

    @property
    def size(self) -> int:
        return min([x.accessible_size for x in self._iters])

    @property
    def input_order(self) -> List:
        return self._input_order

    @property
    def iters_count(self) -> int:
        return self._iters_count

    @abstractmethod
    def read(self) -> Union[Dict[str, any], List[Dict[str, any]]]:
        pass

    @abstractmethod
    def close(self):
        raise NotImplementedError


class BlockMapReaderWithOriginData(DataFrameReader):
    """
    Return both op's input data and origin data.
    """

    def __init__(self, input_dfs: dict, input_order: List):
        iters = []
        for _, x in input_dfs.items():
            ite = x['df'].map_iter(block=True)
            iters.append([ite, x['cols']])

        super().__init__(iters, input_order)
        self._lock = threading.Lock()
        self._close = False

    def read(self) -> Tuple[Dict[str, any], Tuple]:
        """
        Read data from dataframe.

        Can read data from multi dataframes or a single one. If reading from multiple,
        shorter df's will fill their values with None.
        """
        if self._close:
            raise StopIteration

        ret = {}
        origin_data = ()
        count_stop = self.iters_count

        with self._lock:
            for x in self._iters:
                try:
                    data = next(x[0])
                except StopIteration:
                    # If stop iteration raised, fill data with None
                    if self._close:
                        raise StopIteration  #pylint: disable=raise-missing-from
                    for (key, index) in x[1]:
                        ret[key] = None
                    origin_data += (None,)
                    count_stop -= 1
                    continue

                if self._close:
                    raise StopIteration
                if data is not None:
                    for (key, index) in x[1]:
                        ret[key] = data[index].value
                    origin_data += data

            if count_stop == 0:
                raise StopIteration
            # TODO: origin_data ordering with filter
            return ret, origin_data

    def close(self):
        self._close = True
        for x in self._iters:
            x[0].notify()


class BlockMapDataFrameReader(BlockMapReaderWithOriginData):
    """
    Map dataframe reader.
    """

    def read(self) -> Dict[str, any]:
        output, _ = super().read()
        return output


class BatchFrameReader(DataFrameReader):
    """
    Batch reader.

    Reads batches of data from one or multiple dataframes. Missing values will be
    replaced with none.
    """

    def __init__(self, input_dfs: dict, input_order: List, batch_size: int, step: int):
        assert batch_size >= 1 and step >= 1
        iters = []
        for _, x in input_dfs.items():
            ite = x['df'].batch_iter(batch_size, step, block=True)
            iters.append([ite, x['cols']])

        super().__init__(iters, input_order)
        self._lock = threading.Lock()
        self._close = False

    def read(self) -> List[Dict[str, any]]:
        if self._close:
            raise StopIteration

        ret = {}
        count_stop = self.iters_count

        with self._lock:
            for x in self._iters:
                try:
                    data = next(x[0])
                except StopIteration:
                    count_stop -= 1
                    if self._close:
                        raise StopIteration #pylint: disable=raise-missing-from
                    continue

                if self._close:
                    raise StopIteration
                if data is not None:
                    for i, row in enumerate(data):
                        if i not in ret:
                            ret[i] = {}
                        for (key, index) in x[1]:
                            ret[i][key] = row[index].value

            if count_stop == 0:
                raise StopIteration

            # Fill missing values with Nones
            # TODO: Figure out faster/better logic for this
            for key, val in ret.items():
                for x in self.input_order:
                    if x not in val:
                        val[x] = None

            res = []
            for i in range(len(ret)):
                res.append(namedtuple('input', ret[i].keys())(**ret[i]))
            return res

    def close(self):
        self._close = True
        for x in self._iters:
            x[0].notify()
