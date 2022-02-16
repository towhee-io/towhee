
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

from typing import Dict, List, Optional

from towhee.engine.operator_io import reader as io_reader
from towhee.engine.operator_io import writer as io_writer
from towhee.dataframe import DataFrame


def create_reader(
    df: DataFrame,
    iter_type: str,
    inputs_index: Dict[str, int],
    iter_params: Optional[Dict] = None
) -> io_reader.DataFrameReader:
    if iter_type.lower() in ['map', 'flatmap', 'concat', 'filter', 'generator']:
        return io_reader.BlockMapReaderWithOriginData(df, inputs_index)
    elif iter_type.lower() == 'window':
        return io_reader.BatchFrameReader(df, inputs_index, **iter_params)
    elif iter_type.lower() == 'time_window':
        return io_reader.TimeWindowReader(df, inputs_index, **iter_params)
    else:
        raise NameError('Can not find %s iters' % iter_type)


def create_writer(iter_type: str, outputs: List[DataFrame]) -> io_writer.DataFrameWriter:
    assert len(outputs) == 1
    if iter_type.lower() in ['map', 'flatmap', 'concat', 'window', 'generator', 'time_window']:
        return io_writer.DictDataFrameWriter(outputs[0])
    elif iter_type.lower() == 'filter':
        return io_writer.RowDataFrameWriter(outputs[0])
    else:
        raise NameError('Can not find %s iters' % iter_type)
