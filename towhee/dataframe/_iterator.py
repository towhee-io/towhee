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


from towhee.dataframe.dataframe import DataFrame


class ScalarIterator:
    """
    Traverse the variable set with one row per move
    """

    def __init__(self, var_set: VariableSet):
        """
        Args:
            var_set: the variable set to loop through.
        """
        self._var_set = var_set

    def __iter__(self):
        raise NotImplementedError


class BatchIterator:
    """
    Traverse the variable set with fixed-size-batch per move
    """

    def __init__(self, var_set: VariableSet, size: int = None):
        """
        Args:
            var_set: the variable set to loop through.
            size: batch size. If size is None, then the whole variable set will be
                batched together.
        """
        self._var_set = var_set
        self._batch_size = size

    def __iter__(self):
        """
        Examples:
        """
        raise NotImplementedError


class GroupIterator:
    """
    Traverse the variable set based on a custom group-by function
    """

    def __init__(self, var_set: VariableSet, group_by_func: function):
        """
        Args:
            var_set: the variable set to loop through.
            group_by_func: the group by function.
        """
        self._var_set = var_set
        self._group_by_func = group_by_func

    def __iter__(self):
        """
        Examples:
        """
        raise NotImplementedError


class RepeatIterator:
    """
    In the case that a variable set has only one row, *RepeatIterator* will repeatly
    access this row.
    """

    def __init__(self, var_set: VariableSet, n: int = None):
        """
        Args:
            var_set: the variable set to loop through.
            n: the repeat times. If not set, the iterator will never ends.
        """
        self._var_set = var_set
        self._n = n

    def __iter__(self):
        """
        Examples:
        """
        raise NotImplementedError

