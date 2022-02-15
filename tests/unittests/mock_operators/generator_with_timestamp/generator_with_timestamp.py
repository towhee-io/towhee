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


from typing import Generator, NamedTuple
from towhee.operator import Operator, SharedType


Outputs = NamedTuple("Outputs", [("num", int), ("TIMESTAMP", int)])


class GeneratorWithTimestamp(Operator):
    """
    Generator operator.
    """

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        pass

    def __call__(self, num: int) -> Generator:
        self._num = 0
        while True:
            if self._num >= num:
                break
            else:
                yield Outputs(self._num, self._num * 500)
                self._num += 1

    @property
    def shared_type(self):
        return SharedType.NotShareable
