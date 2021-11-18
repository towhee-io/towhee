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


import towhee.engine.operator_io.reader as io_reader
import towhee.engine.operator_io.writer as io_writer
from towhee.dataframe import DataFrame
from typing import Dict, List


def create_reader(
    input_dfs: dict,
    iter_type: str,
) -> io_reader.DataFrameReader:

    if iter_type.lower() in ["map"]:
        return io_reader.BlockMapDataFrameReader(input_dfs)
    else:
        raise NameError("Can not find %s iters" % iter_type)


def create_writer(iter_type: str, outputs: List[DataFrame]) -> io_writer.DataFrameWriter:
    assert len(outputs) == 1
    if iter_type.lower() == "map":
        return io_writer.DataFrameWriter(outputs[0])
    else:
        raise NameError("Can not find %s iters" % iter_type)
