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


class _Schema:
    """
    Schema of dataframe.

    Every dataframe has _Frame cols.
    """

    def __init__(self):
        self._cols = []
        self._key_index = {}

    def add_col(self, name: str, col_type: str) -> bool:
        if name in self._key_index:
            return False

        self._cols.append((name, col_type))
        self._key_index[name] = len(self._cols) - 1
        return True

    def col_index(self, col_name: str) -> int:
        return self._key_index[col_name]

    def col_key(self, index: int) -> str:
        return self._cols[index][0]

    def col_type(self, index: int):
        return self._cols[index][1]

    @property
    def col_count(self):
        return len(self._cols)

    @property
    def cols(self):
        return self._cols
