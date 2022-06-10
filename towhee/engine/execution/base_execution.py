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

from towhee.utils.log import engine_log


class BaseExecution:
    """
    Execute an operator
    """

    def __call__(self, *arg, **kws):
        if self._func is not None:
            try:
                return self._func(*arg)
            except Exception as e: # pylint: disable=broad-except
                engine_log.info('%s cannot compile with numba with error: %s', self._name, e)
        self.__check_init__()
        if bool(self._index):
            res = self.__apply__(*arg, **kws)

            # Multi outputs.
            if isinstance(res, tuple):
                if not isinstance(self._index[1],
                                  tuple) or len(self._index[1]) != len(res):
                    raise IndexError(
                        f'Op has {len(res)} outputs, but {len(self._index[1])} indices are given.'
                    )
                for i, j in zip(self._index[1], res):
                    setattr(arg[0], i, j)
            # Single output.
            else:
                setattr(arg[0], self._index[1], res)
            return arg[0]
        else:
            return self._op(*arg, **kws)
