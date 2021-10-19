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

from towhee.dataframe import DataFrame, Variable, DataFrameIterator
from typing import Dict, Optional, Tuple, Union, List, NamedTuple


class DataFrameWriter:
    """
    Df writer
    """

    def __init__(self, output_df: DataFrame):
        self._output_df = output_df

    def write(self, output_data: NamedTuple) -> bool:

        # TODO (jiangjunjie) deal with task exception
        if output_data is not None:
            return self._output_df.put_dict(output_data._asdict())

    def close(self):
        self._output_df.seal()


class DataFrameReader(ABC):
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

    def _to_op_inputs(self, cols: Tuple[Variable]) -> Dict[str, any]:
        """
        Read from cols, combine op inputs
        """
        ret = {}
        for key, index in self._op_inputs_index.items():
            ret[key] = cols[index].value
        return ret


class MapDataFrameReader(DataFrameReader):
    """
    Map dataframe reader
    """

    def __init__(
        self,
        input_df: DataFrame,
        op_inputs_index: Dict[str, int]
    ):
        super().__init__(input_df.map_iter(), op_inputs_index)

    def read(self) -> Optional[Dict[str, any]]:
        """
        Read data from dataframe, get cols by operator_repr info
        """
        try:
            data = next(self._iter)
            if not data:
                return {}
            return self._to_op_inputs(data[0])

        except StopIteration:
            return None


def create_reader(
    inputs: List[DataFrame],
    iter_type: str,
    inputs_index: Dict[str, int]
) -> DataFrameReader:

    if iter_type.lower() == "map":
        assert len(inputs) == 1, "%s iter takes one dataframe, but %s dataframes are found" % (
            iter_type, len(inputs)
        )
        return MapDataFrameReader(inputs[0], inputs_index)
    else:
        raise NameError("Can not find %s iters" % iter_type)


def create_writer(outputs: List[DataFrame]) -> DataFrameWriter:
    assert len(outputs) == 1
    return DataFrameWriter(outputs[0])
