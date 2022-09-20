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


import threading
from typing import List


class ColStorage:
    def __init__(self, schema_info):
        self._data = []
        self._schema = Schema(schema_info)

    def put(self, inputs: List):
        assert len(inputs) = self._schema.size
        for i in range(len(inputs)):
            self._data[i].put(inputs[i])


class Schema:
    def __init__(self, schema_info):
        pass

    @property
    def size(self):
        pass

class Column:
    def __init__(self):
        pass
