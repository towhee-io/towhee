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

from typing import List
from pydantic import BaseModel

from towhee.runtime.data_queue import ColumnType
from towhee.runtime.constants import (
    WindowAllConst,
    WindowConst,
    ReduceConst,
    FilterConst,
    TimeWindowConst,
    FlatMapConst,
    ConcatConst,
    MapConst
)


# pylint: disable=redefined-builtin
class SchemaRepr(BaseModel):
    """
    A `SchemaRepr` represents the data queue schema.

    Args:
        name (`str`): The name column data.
        type (`ColumnType`): The type of the column data, such as ColumnType.SCALAR or ColumnType.QUEUE.
    """
    name: str
    type: ColumnType

    @staticmethod
    def from_dag(col_name: str, iter_type: str, inputs_type: List = None):
        """Return a SchemaRepr from the dag info.

        Args:
            col_name (`str`): Schema name.
            iter_type (`Dict[str, Any]`): The iteration type of this node.
            inputs_type (`List`): A list of the inputs schema type.

        Returns:
            SchemaRepr object.
        """
        if iter_type in [FlatMapConst.name, WindowConst.name, TimeWindowConst.name]:
            col_type = ColumnType.QUEUE
        elif iter_type == ConcatConst.name:
            col_type = inputs_type[0]
        elif iter_type in [WindowAllConst.name, ReduceConst.name]:
            col_type = ColumnType.SCALAR
        elif iter_type in [MapConst.name, FilterConst.name]:
            if inputs_type is not None and ColumnType.QUEUE in inputs_type:
                col_type = ColumnType.QUEUE
            else:
                col_type = ColumnType.SCALAR
        else:
            raise ValueError(f'Unknown iteration type: {iter_type}')
        return SchemaRepr(name=col_name, type=col_type)
