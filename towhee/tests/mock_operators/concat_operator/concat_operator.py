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


from typing import NamedTuple
from collections import defaultdict
from towhee.operator import Operator


class ConcatOperator(Operator):
    """
    Stateful concat operator that can perform scalar and nonscalar concat. If no new data
    is presented, the row will be skipped.
    """

    def __init__(self, repeat_end: bool) -> None:
        super().__init__()
        self._scalar = repeat_end
        self._last_seens = defaultdict(lambda: None)

    def __call__(self, **kwargs):
        ret = []
        for col, val in kwargs.items():
            if val is None:
                if self._scalar:
                    ret.append(self._last_seens[col])
                else:
                    ret.append(None)
            else:
                ret.append(val)
                self._last_seens[col] = val

        #If no new data added, dont return anything.
        if not set(kwargs.values()) - {None}:
            pass
        else:
            cols = [(str(col), type(val)) for col, val in kwargs.items()]
            output = NamedTuple("Outputs", cols)
            return output(*ret) #pylint: disable=not-callable
