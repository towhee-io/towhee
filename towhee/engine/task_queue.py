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


from towhee.engine.task import Task


class TaskQueue:
    """
    The queue where scheduler push tasks and executor pop tasks.
<<<<<<< HEAD
    Each TaskExecutor has one TaskQueue.
    """

=======
    Each device has one TaskQueue.
    """

    def __init__(self)

>>>>>>> 9ada236dd1ea0de151ebd569a018da7ffce5cac7
    @property
    def empty(self) -> bool:
        """
        Indicator whether TaskQueue is empty.
        True if the queue has no tasks.
        """
        raise NotImplementedError
    
    @property
    def size(self) -> int:
        """
        Number of tasks in the TaskQueue.
        """
        raise NotImplementedError

    def push(self, task: Task) -> None:
        raise NotImplementedError

    def pop(self) -> Task:
        raise NotImplementedError