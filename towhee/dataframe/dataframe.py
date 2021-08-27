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


from towhee.dataframe._iterator import ScalarIterator, GroupIterator, BatchIterator, RepeatIterator


class DataFrame:
    """
    A collection of immutable, potentially heterogeneous arrays of data.
    """

    def __init__(self, name: str = None, data = None):
        """
        Args:
            name: the name of the DataFrame
            data: a list of Array with same size
        """
        self.name = name
        self.iter = DFIterator(self)
        self.columns = data

    def __iter__(self):
        return iter(self.iter)



class DFIterator:
    """
    The DataFrame iterator
    """
    def __init__(self, df: DataFrame):
        self._iter = ScalarIterator(df)
        self._df = df

    def __iter__(self):
        return iter(self._iter)
    
    def group_by(self, func):
        self._iter = GroupIterator(self._df, func)
    
    def batch(self, size = None):
        self._iter = BatchIterator(self._df, size)

    def repeat(self, n = None):
        self._iter = RepeatIterator(self._df, n)