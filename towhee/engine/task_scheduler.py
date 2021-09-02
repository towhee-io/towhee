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


from abc import ABC, abstractmethod
from towhee.engine.task import Task
from towhee.engine.engine import Engine


class TaskScheduler(ABC):
    """
    Task scheduler interface
    """

    @abstractmethod
    def on_task_ready(self, task: Task):
        pass

    @abstractmethod
    def on_task_start(self, task: Task):
        pass

    @abstractmethod
    def on_task_finish(self, task: Task):
        pass


class FIFOTaskScheduler(TaskScheduler):
    """
    Scheduling tasks in a first in first out manner
    """

    def __init__(self, engine: Engine):
        self._engine = engine
        self._queue = []

    def on_task_ready(self, task: Task):
        pass

    def on_task_start(self, task: Task):
        pass

    def on_task_finish(self, task: Task):
        pass