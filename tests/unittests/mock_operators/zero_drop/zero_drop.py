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


from towhee.operator import Operator, SharedType


class ZeroDrop(Operator):
    """
    Drop zero.
    """

    def __init__(self):  # pylint: disable=super-init-not-called
        pass

    def __call__(self, num: int) -> bool:
        if not isinstance(num, int):
            raise RuntimeError(f'Input {num} is not int type')
        return num != 0

    @property
    def shared_type(self):
        return SharedType.Shareable
