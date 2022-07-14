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
from typing import Iterator

from towhee.functional.mixins.dag import register_dag


class StreamMixin:
    """
    Stream related mixins.
    """
    @register_dag
    def stream(self):
        """
        Create a stream data collection.

        Examples:
        1. Convert a data collection to streamed version

        >>> from towhee import DataCollection
        >>> dc = DataCollection([0, 1, 2, 3, 4])
        >>> dc.is_stream
        False

        >>> dc = dc.stream()
        >>> dc.is_stream
        True
        """
        # pylint: disable=protected-access
        iterable = iter(self._iterable) if not self.is_stream else self._iterable
        return self._factory(iterable, parent_stream=False)

    @register_dag
    def unstream(self):
        """
        Create a unstream data collection.

        Examples:

        1. Create a unstream data collection

        >>> from towhee import DataCollection
        >>> dc = DataCollection(iter(range(5))).unstream()
        >>> dc.is_stream
        False

        2. Convert a streamed data collection to unstream version

        >>> dc = DataCollection(iter(range(5)))
        >>> dc.is_stream
        True
        >>> dc = dc.unstream()
        >>> dc.is_stream
        False
        """
        iterable = list(self._iterable) if self.is_stream else self._iterable
        return self._factory(iterable, parent_stream=False)

    @property
    def is_stream(self):
        """
        Check whether the data collection is stream or unstream.

        Examples:

        >>> from towhee import DataCollection
        >>> from typing import Iterable
        >>> dc = DataCollection([0,1,2,3,4])
        >>> dc.is_stream
        False

        >>> result = dc.map(lambda x: x+1)
        >>> result.is_stream
        False
        >>> result._iterable
        [1, 2, 3, 4, 5]

        >>> dc = DataCollection(iter(range(5)))
        >>> dc.is_stream
        True

        >>> result = dc.map(lambda x: x+1)
        >>> result.is_stream
        True
        >>> isinstance(result._iterable, Iterable)
        True

        """
        return isinstance(self._iterable, Iterator)
