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


class Engine:
    """
    """

    def __init__(self, config: dict):
        """
        Args:
            config: the Engine configurations. 
        """
        self._config = config
        self.on_task_ready_handlers = []
        self.on_task_start_handlers = []
        self.on_task_finish_handlers = []

        # self._setup_task_executors()
        # self._setup_task_scheduler()

        raise NotImplementedError


    def add_pipeline(self, pipeline):
        pipeline.on_task_ready_handlers.append(self.on_task_ready_handlers)
        pipeline.on_task_start_handlers.append(self.on_task_start_handlers)
        pipeline.on_task_finish_handlers.append(self.on_task_finish_handlers)
        raise NotImplementedError

    def _setup_task_executors(self):
        """
        Create TaskExecutors to manage task execution on CPU and GPU devices.
        """
        # todo: device scan
        # todo: parse engine configs
        # todo: create a TaskExecutor on each of the devices
        raise NotImplementedError
    
    def _setup_task_scheduler(self):
        # todo: parse engine configs
        # todo: create TaskScheduler
        # self._task_scheduler(self)
        raise NotImplementedError