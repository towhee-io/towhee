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


from queue import Queue

from towhee.engine.operator_io.reader import ReaderBase

class StopFrame:
    pass


class MockReader(ReaderBase):
    """
    MockReader for engine test
    """
    def __init__(self, queue: Queue):
        self._queue = queue

    def read(self):
        # Blocking if queue is empty
        data = self._queue.get()

        if not isinstance(data, StopFrame):
            return data
        else:
            raise StopIteration()

    def close(self):
        self._queue.put(StopFrame())
