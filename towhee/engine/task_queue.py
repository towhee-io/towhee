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


import queue

from towhee.engine.task import Task


class TaskQueue:
    """
    A queue for `Task` objects exposed to the `TaskScheduler` which allows the scheduler
    to assign tasks to individual instances of `TaskExecutor`.

    Args:
        size:
            Maximum number of allowable queue elements.
    """

    def __init__(self, size: int = 0):
        self._queue = queue.Queue(size)

    @property
    def empty(self) -> bool:
        """Indicates whether `TaskQueue` is empty. Returns true if the queue has no
        tasks.
        """
        return self._queue.empty()

    @property
    def full(self) -> bool:
        """Indicates whether or not the `TaskQueue` is at its maximum capacity.
        """
        return self._queue.full()

    @property
    def size(self) -> int:
        """Returns the number of tasks in the TaskQueue.
        """
        return self._queue.qsize()

    def push(self, task: Task) -> bool:
        """Pushes a `Task` object to the end of the queue. Returns `True` if the
        operation was successful and `False` otherwise. A return value of `False` most
        likely indicates that the queue has reached its maximum capacity.

        Args:
            task: (`towhee.engine.Task`)
                `Task` object to add to the end of the queue.
        """
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            return False
        return True

    def pop(self) -> Task:
        """Attempts to acquire the first item off of the queue.

        Returns:
            (`towhee.engine.Task`)
                First `Task` object available on the queue, or None if the queue is
                empty
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
