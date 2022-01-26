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


class ConcatOperator(Operator):
    """
    Concat operator.

    Combine multiple dataframes into one. It's an internal operator, and the
    framework will set readers and writer to it, so the operator can use readers
    and wirter directly.

    Row based concat:

    Col based concat:
    """

    def __init__(self, concat_type: str = 'row') -> None:
        if concat_type.lower() == 'row':
            self._concat_func = self._row_based_concat
        elif concat_type.lower() == 'col':
            self._concat_func = self._col_based_concat
        else:
            raise RuntimeError('Unkown concat type: %s ' % concat_type)

    @staticmethod
    def _read(reader):
        try:
            data, _ = reader.read()
            return False, data
        except StopIteration:
            return True, None
        pass

    def _row_based_concat(self):
        while True:
            result = {}
            for reader in self._readers:
                is_end, data = ConcatOperator._read(reader)
                if not is_end:
                    result.update(data)
            if len(result) == 0:
                break
            self._writer.write(result)

    def _col_based_concat(self):
        raise NotImplementedError

    def __call__(self) -> bool:
        self._concat_func()

    @property
    def shared_type(self):
        return SharedType.NotReusable
