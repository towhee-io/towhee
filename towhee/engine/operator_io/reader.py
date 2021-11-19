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

# from towhee.dataframe import DataFrame, Variable, DataFrameIterator
from typing import Dict, Optional, Union, List


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

    def __init__(self, input_dfs: dict):
        self._iters = []
        for _, x in input_dfs.items():
            ite = x['df'].map_iter()
            self._iters.append([ite, x['cols']])
        self._iters_count = len(self._iters)


    @abstractmethod
    def read(self) -> Union[Dict[str, any], List[Dict[str, any]]]:
        pass

    @property
    def size(self) -> int:
        return min([x.accessible_size for x in self._iters])

    @abstractmethod
    def close(self):
        raise NotImplementedError

    # def _to_op_inputs(self, cols: Tuple[Variable]) -> Dict[str, any]:
    #     """
    #     Read from cols, combine op inputs
    #     """
    #     ret = {}
    #     for key, index in self._op_inputs_index.items():
    #         ret[key] = cols[index].value
    #     return ret


class BlockMapDataFrameReader(DataFrameReader):
    """
    Map dataframe reader
    """

    def __init__(
        self,
        input_dfs: dict,
    ):
        super().__init__(input_dfs)
        self._lock = threading.Lock()
        self._close = False

    def read(self) -> Optional[Dict[str, any]]:
        """
        Read data from dataframe, get cols by operator_repr info
        """
        if self._close:
            raise StopIteration
        ret = {}
        bitmap = list(range(self._iters_count))
        with self._lock:
            while len(bitmap) > 0:
                remove_index = set()
                for bit_index, iter_index in enumerate(bitmap):
                    if self._close:
                        raise StopIteration
                    # self._iters[x][0] is the iterator
                    # self._iters[x][1] are the columns for that iterator
                    data = next(self._iters[iter_index][0])
                    print("data being read in: ", data, "from: ",  self._iters[iter_index][0]._df_name)
                    if data == None:
                        print("no present data, continuing to wait")
                    if data is not None:
                        for (key, index) in self._iters[iter_index][1]:
                            if data[index] is None:
                                print("present data holder is: ", None)
                            elif data[index].value is None:
                                print("present data is: ", None)
                            else:
                                print("present data is: ", data[index].value)
                            ret[key] = data[index].value
                        print("Removing bitmask values: ", (bit_index, iter_index) )
                        remove_index.add(bit_index)
                    
                # enumerating set is fastest way to remove multiple indexes
                bitmap = [i for j, i in enumerate(remove_index) if j not in remove_index]
            print(ret)
            return ret

        # # Previous
        # if self._close:
        #     raise StopIteration

        # with self._lock:
        #     data = next(self._iter)
        #     if self._close:
        #         raise StopIteration

        #     if not data:
        #         return {}
        #     return self._to_op_inputs(data)

    def close(self):
        self._close = True
        for x in self._iters:
            x[0].notify()

# class MultiMapDataFrameReader(MultiDataFrameReader):
#     """
#     Map dataframe reader for multiple inputs.
#     """

#     def __init__(
#         self,
#         inputs: DataFrame,
#     ):
#         iter_list = []
#         for _, x in inputs.items():
#             ite = x['df'].map_iter()
#             iter_list.append((ite, x['cols']))

#         super().__init__(iter_list)

#     def read(self) -> Optional[Dict[str, any]]:
#         """
#         Read data from dataframe, get cols by operator_repr info

#         """
#         # TODO: Reader needs to be thread safe

#         ret = {}
#         count_stop = 0
#         # ite[0] is the iterator
#         # ite[1] is the columns for that iterator
#         for ite in self._iters:
#             try:
#                 data = next(ite[0])
#                 if data is not None:
#                     # key is operator parameter
#                     # index is column
#                     for (key, index) in ite[1]:
#                         ret[key] = data[index][0].value
#                 else:
#                     for (key, index) in ite[1]:
#                         ret[key] = None
#             # will StopIteration be thrown after
#             except StopIteration:
#                 for key, index in ite[1]:
#                     ret[key] = None
#                 count_stop += 1

#         if count_stop == self._size:
#             return None
#         return ret
