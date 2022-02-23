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

from towhee.dataframe.dataframe import DataFrame
from towhee.dataframe.iterators import DataFrameIterator, MapIterator, BatchIterator, WindowIterator


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

    def __init__(self, it: DataFrameIterator, op_inputs_index: Dict[str, int]):
        self._op_inputs_index = op_inputs_index
        self._iter = it

    @abstractmethod
    def read(self) -> Union[Dict[str, any], List[Dict[str, any]]]:
        pass

    @property
    def size(self) -> int:
        return self._iter.accessible_size

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def _to_op_inputs(self, cols: Tuple) -> Dict[str, any]:
        """
        Read from cols, combine op inputs
        """
        ret = {}
        for key, index in self._op_inputs_index.items():
            ret[key] = cols[index]
        return ret


class BlockMapReaderWithOriginData(DataFrameReader):
    """
    Return both op's input data and origin data.
    """

    def __init__(
        self,
        input_df: DataFrame,
        op_inputs_index: Dict[str, int]
    ):
        super().__init__(MapIterator(input_df, True), op_inputs_index)
        self._lock = threading.Lock()
        self._close = False

    def read(self) -> Tuple[Dict[str, any], Tuple]:
        """
        Read data from dataframe, get cols by operator_repr info
        """
        if self._close:
            raise StopIteration

        with self._lock:
            data = next(self._iter)
            # map and batch iterators both return list of tuples
            data = data[0]
            if self._close:
                raise StopIteration

            if not data:
                return {}, ()
            return self._to_op_inputs(data), data

    def close(self):
        self._close = True
        self._iter.notify()


class BatchFrameReader(DataFrameReader):
    """
    Batch reader.
    """

    def __init__(self, input_df: DataFrame, op_inputs_index: Dict[str, int],
                 batch_size: int, step: int):
        assert batch_size >= 1 and step >= 1
        super().__init__(BatchIterator(input_df, batch_size, step, True), op_inputs_index)
        self._close = False
        self._lock = threading.Lock()

    def read(self) -> List[Dict[str, any]]:
        if self._close:
            raise StopIteration

        with self._lock:
            data = next(self._iter)
            if self._close:
                raise StopIteration

            if not data:
                return []
            else:
                res = []
                for row in data:
                    data_dict = self._to_op_inputs(row)
                    res.append(namedtuple('input', data_dict.keys())(**data_dict))
                return res

    def close(self):
        self._close = True
        self._iter.notify()


class TimeWindowReader(DataFrameReader):
    """
    Time window reader
    """

    def __init__(
        self,
        input_df: DataFrame,
        op_inputs_index: Dict[str, int],
        time_range_sec: int,
        time_step_sec: int
    ):
        super().__init__(WindowIterator(input_df, start = 0,
                                                window_size = time_range_sec,
                                                step = time_step_sec,
                                                use_timestamp = True,
                                                block = True), op_inputs_index)
        self._lock = threading.Lock()
        self._close = False

    def _format_to_namedtuple(self, rows):
        ret = []
        for row in rows:
            data_dict = self._to_op_inputs(row)
            ret.append(namedtuple('input', data_dict.keys())(**data_dict))
        return ret

    def read(self) -> List[Dict[str, any]]:
        if self._close:
            raise StopIteration

        with self._lock:
            data = next(self._iter)
            if self._close:
                raise StopIteration

            if not data:
                return []
            else:
                res = []
                for row in data:
                    data_dict = self._to_op_inputs(row)
                    res.append(namedtuple('input', data_dict.keys())(**data_dict))
                return res

    def close(self):
        self._close = True
        self._iter.notify()
