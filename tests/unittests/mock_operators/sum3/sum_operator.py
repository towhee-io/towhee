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

from towhee.operator import Operator, SharedType


class SumOperator(Operator):
    """
    Flat input list.
    """

    def __init__(self):
        pass

    def __call__(self, num1: List[int], num2: List[int]):
        assert len(num1) == len(num2)
        return [i + j for i, j in zip(num1, num2)]

    @property
    def shared_type(self):
        return SharedType.Shareable
