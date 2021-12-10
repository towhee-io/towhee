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


class _ArrayRef:
    """
    `_ArrayRef` maintains `DataFrame` to `Array` references.
    """

    def __init__(self):
        self._offsets = {}
        self._ref_id_allocator = 0

    def add_reader(self) -> int:
        """
        Add a read reference to this `Array`

        Returns:
            (`int`)
                reference id
        """
        ref_id = self._ref_id_allocator
        self._offsets[ref_id] = 0
        self._ref_id_allocator += 1
        return ref_id

    def remove_reader(self, ref_id: int):
        """
        Remove a read reference from this `Array`

        Args:
            ref_id (`int`):
                The reference ID
        """
        self._offsets.pop(ref_id, None)

    def update_reader_offset(self, ref_id: int, offset: int):
        if ref_id in self._offsets:
            self._offsets[ref_id] = offset
        else:
            raise KeyError('reference id %d not fount' % (ref_id))

    def get_reader_offset(self, reader_id):
        return self._offsets[reader_id]
    

    @property
    def min_reader_offsets(self):
        if self._offsets:
            return min(self._offsets.values())
        else:
            return 0
    

    
