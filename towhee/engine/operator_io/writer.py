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
from typing import Tuple, Dict

from towhee.dataframe import DataFrame
from towhee.utils.log import engine_log


class DataFrameWriter(ABC):
    """
    Dataframe writer base.
    """

    def __init__(self, output_df: DataFrame):
        self._output_df = output_df

    def write(self, output_data: any) -> None:
        if output_data is not None:
            self._write(output_data)
        else:
            engine_log.debug('Ignore None data which try to write to dataframe %s',
                             self._output_df.name)

    @abstractmethod
    def _write(self, output_data: any):
        raise NotImplementedError

    def close(self):
        self._output_df.seal()

class DictDataFrameWriter(DataFrameWriter):
    """
    Df writer

    Write dict to the next dataframe.
    """

    def _write(self, output_data: Dict[str, any]) -> None:
        self._output_df.put(output_data)


class RowDataFrameWriter(DataFrameWriter):
    """
    Filter dataframe writer.

    Write prev dataframe's one row to the next dataframe.
    """

    def _write(self, output_data: Tuple) -> None:
        self._output_df.put(output_data)
