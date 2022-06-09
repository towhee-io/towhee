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


class PandasExecution:
    """
    Execute operator on pandas DataFrame
    """

    def __apply__(self, *arg, **kws):
        if isinstance(self._index[0], tuple):  # Multi inputs.
            args = [getattr(arg[0], x) for x in self._index[0]]
        else:  # Single input.
            args = [getattr(arg[0], self._index[0])]
        return self._op(*args, **kws)

    def __dataframe_apply__(self, df):
        self.__check_init__()
        if isinstance(self._index[1], tuple):
            df[list(self._index[1])] = df.apply(self.__apply__,
                                                axis=1,
                                                result_type='expand')
        else:
            df[self._index[1]] = df.apply(self.__apply__, axis=1)
        return df

    def __dataframe_filter__(self, df):
        self.__check_init__()
        return df[self.__apply__(df)]
