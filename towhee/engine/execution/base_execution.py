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


class BaseExecution:
    """
    Basic execution of an operator.
    """
    def __apply__(self, *arg, **kws):
        # Multi inputs.
        if isinstance(self._index[0], tuple):
            args = [getattr(arg[0], x) for x in self._index[0]]
        # Single input.
        else:
            args = [getattr(arg[0], self._index[0])]
        return self._op(*args, **kws)

    def __call__(self, *arg, **kws):
        self.__check_init__()
        if bool(self._index):
            res = self.__apply__(*arg, **kws)

            # Multi outputs.
            if res is not None and len(self._index) == 2:
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
            res = self._op(*arg, **kws)
            return res
