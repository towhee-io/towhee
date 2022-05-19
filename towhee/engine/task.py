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


class DataElement:
    """
    The base class of data element. Each `DataElement` holds a reference of a physical data.
    This class is introduced to avoid unnecessary data copy between driver program end engine.
    """

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class TaskInfo:
    """
    Records the basic infomation of a task. The ID assignment of consecutive task is
    monotonically increasing.

    Args:
        task_id: (`int`)
            The task ID.
        op: (`Operator`)
            The operator
        inputs: (`DataElement`)
            The input data elements
        outputs: (`DataElement`)
            The output data elements
    """

    def __init__(self, task_id, op, inputs: List[DataElement] = None, outputs: List[DataElement] = None):
        self.task_id = task_id
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
