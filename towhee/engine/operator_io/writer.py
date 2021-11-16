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


from towhee.dataframe import DataFrame
from typing import NamedTuple


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
