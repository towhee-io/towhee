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


import threading
from queue import Queue
from typing import Optional


from towhee.engine.runtime_graph import Node


class TaskQueue:
    """Has multi executors, task queue, 
       executor get task from queue
    """
    def put(self, node: Optional(Node)) -> None:
        raise NotImplementedError


class TaskExecutor(threading.Thread):
    """Executive layer
    """
    def __init__(self, name: str, queue: Queue):
        threading.Thread.__init__(self)
        self._name = name
        self._need_stop = False
        self._task_queue = queue
        self.count = 0

    def run(self):
        """Get task from queue, call run function

            while not self._need_stop:
                node = self._task_queue.get()
                if node is None:
                    break
                node.run()
        """
        raise NotImplementedError
