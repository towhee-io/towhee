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

from towhee.dataframe import DataFrame, Variable, DataFrameIterator
from typing import Dict, Optional, Tuple, Union, List, NamedTuple


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

    def _to_op_inputs(self, cols: Tuple[Variable]) -> Dict[str, any]:
        """
        Read from cols, combine op inputs
        """
        ret = {}
        for key, index in self._op_inputs_index.items():
            ret[key] = cols[index].value
        return ret


class BlockMapDataFrameReader(DataFrameReader):
    """
    Map dataframe reader
    """

    def __init__(
        self,
        input_df: DataFrame,
        op_inputs_index: Dict[str, int]
    ):
        super().__init__(input_df.map_iter(True), op_inputs_index)
        self._lock = threading.Lock()
        self._close = False

    def read(self) -> Optional[Dict[str, any]]:
        """
        Read data from dataframe, get cols by operator_repr info
        """
        try:
            if self._close:
                return None

            with self._lock:
                data = next(self._iter)
                if self._close:
                    return None
                
                if not data:
                    return {}
                return self._to_op_inputs(data[0])

        except StopIteration:
            return None

    def close(self):
        self._close = True
        self._iter.notify()
