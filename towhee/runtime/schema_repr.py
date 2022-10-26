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

from towhee.runtime.data_queue import ColumnType
from towhee.utils.log import engine_log


# pylint: disable=redefined-builtin
class SchemaRepr:
    """
    A `SchemaRepr` represents the data queue schema.

    Args:
        name (`str`): The name column data.
        type (`ColumnType`): The type of the column data, such as ColumnType.SCALAR or ColumnType.QUEUE.
    """
    def __init__(self, name: str, type: 'ColumnType'):
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> 'ColumnType':
        return self._type

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
        if iter_type in ['flat_map', 'window', 'window_all', 'time_window', 'concat']:
            col_type = ColumnType.QUEUE
        elif inputs_type is None:
            col_type = ColumnType.SCALAR
        elif iter_type in ['map', 'filter']:
            col_type = ColumnType.SCALAR
            if ColumnType.QUEUE in inputs_type:
                col_type = ColumnType.QUEUE
        else:
            engine_log.error('Unknown iteration type: %s', iter_type)
            raise ValueError(f'Unknown iteration type: {iter_type}')
        return SchemaRepr(col_name, col_type)
