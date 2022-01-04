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
from towhee.operator import PyOperator, SharedType


class AddOperator(PyOperator):
    """
    Stateful operator
    """
    def __init__(self, factor: int) -> None:
        super().__init__()
        self._factor = factor

    def __call__(self, num: int) -> NamedTuple("Outputs", [("sum", int)]):
        Outputs = NamedTuple("Outputs", [("sum", int)])
        return Outputs(self._factor + num)

    @property
    def shared_type(self):
        return SharedType.Shareable
