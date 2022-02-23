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


from typing import Optional


FRAME = 'frame'


class _Frame:
    """
    Row info of dataframe.
    row_id (`int`):
        Row id of dataframe, an auto-increment data.
    parent_path (`str`):
        FlatMap's output data, will set the data.

        Example:
        The parent_path changes between flatmap operators.

                   dataframe                  dataframe                    dataframe

                    frame                       frame                       frame
                 ┌───────────┐               ┌───────────┐               ┌───────────┐
           f1    │1  '' None │     f1-1      │1   1  None│       f1-1-1  │1  1-1 None│
                 ├───────────┤               ├───────────┤               ├───────────┤
           f2    │2  '' None │     f1-2      │2   1  None│       f1-1-2  │2  1-1 None│
                 ├───────────┤               ├───────────┤               ├───────────┤
                 │           │     f2-3      │3   2  None│       f1-2-3  │3  1-2 None│
                 ├───────────┤               ├───────────┤               ├───────────┤
                 │           │               │           │       f2-3-4  │4  2-3 None│
                 │           │               │           │               │           │
                 └───────────┘               └───────────┘               └───────────┘
                                 ┌───────────┐           ┌───────────┐
                                 │           │           │           │
                ───────────────► │  FlatMap  ├──────────►│  FlatMap  ├────────────────►
                                 │           │           │           │
                                 └───────────┘           └───────────┘


    timestamp (`int`):
        Video images will have timestamp attribute, the video-decoder will add the info to frames.
    empty(`bool`):
        Filter-operator will set empty = True is this frame is filtered.
    prev_id: (`int`):
        Operator input-data's row_id.
    """

    def __init__(self, row_id: int = -1, parent_path: str = '',
                 timestamp: int = Optional[None], empty: bool = False,
                 prev_id=-1):
        self._row_id = row_id
        self._timestamp = timestamp
        self._parent_path = parent_path
        self._empty = empty
        self._prev_id = prev_id

    @property
    def row_id(self):
        return self._row_id

    @row_id.setter
    def row_id(self, row_id: int):
        self._row_id = row_id

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp: int):
        self._timestamp = timestamp

    @property
    def parent_path(self):
        return self._parent_path

    @parent_path.setter
    def parent_path(self, parent_path: str):
        self._parent_path = parent_path

    @property
    def prev_id(self):
        return self._prev_id

    @prev_id.setter
    def prev_id(self, prev_id: int):
        self._prev_id = prev_id

    @property
    def empty(self):
        return self._empty

    @empty.setter
    def empty(self, empty: bool):
        self._empty = empty

    def __str__(self) -> str:
        return f'row_id={self.row_id}:parent_path={self.parent_path}:timestamp={self.timestamp}:empty={self.empty}'
