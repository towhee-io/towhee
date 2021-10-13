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

from towhee.engine.singleton import singleton
from towhee.engine.pipeline import Pipeline
from towhee.engine.task_scheduler import FIFOTaskScheduler
from towhee.engine.task_executor import TaskExecutor


@singleton
class EngineConfig:
    """
    Global engine config
    """

    def __init__(self):
        self._sched_type = 'fifo'
        self._cache_path = None
        self._sched_interval_ms = 20

    @property
    def sched_type(self):
        return self._sched_type

    @sched_type.setter
    def sched_type(self, sched_type: str):
        self._sched_type = sched_type

    @property
    def cache_path(self):
        return self._cache_path

    @cache_path.setter
    def cache_path(self, cache_path: str):
        self._cache_path = cache_path

    @property
    def sched_interval_ms(self):
        return self._sched_interval_ms

    @sched_interval_ms.setter
    def sched_interval_ms(self, sched_interval_ms: int):
        self._sched_interval_ms = sched_interval_ms

    def __str__(self):
        return str(self.__dict__)


@singleton
class Engine(threading.Thread):
    """Engines are the core component responsible for deliving results to the user. A
    single engine may be composed of multiple pipelines.
    """

    def __init__(self):
        super().__init__()

        self._config = EngineConfig()
        self._pipelines = []
        self.setDaemon(True)

        # Setup executors and scheduler.
        self._setup_task_execs()
        self._setup_task_sched()

    def stop(self) -> None:
        self._task_sched.stop()
        for task_exec in self._task_execs:
            task_exec.stop()

    def run(self):
        self._task_sched.schedule_forever(self._config.sched_interval_ms)

    def add_pipeline(self, pipeline: Pipeline):
        """Add a single pipeline to this engine. Pipelines can be added long after an
        engine has been instantiated.

        Args:
            pipeline: `towhee.Pipeline`
                A single pipeline with which this engine will be assume execution
                responsibility.
        """
        # pipeline.add_task_ready_handler(self._on_task_ready)
        # pipeline.add_task_start_handler(self._on_task_start)
        # pipeline.add_task_finish_handler(self._on_task_finish)

        pipeline.register(self._task_sched)
        self._pipelines.append(pipeline)

    def _setup_task_execs(self):
        """(Initialization function) Scan for devices and create TaskExecutors to
        manage task execution on CPU, GPU, and other devices.
        """
        self._task_execs = []

        # TODO(fzliu): Perform device scan.
        # Ping (filip-halt) when done
        dev_names = ['cpu:0']

        # Create executor threads and begin running.
        for name in dev_names:
            executor = TaskExecutor(
                name=name, cache_path=self._config.cache_path)
            self._task_execs.append(executor)
            executor.start()

    def _setup_task_sched(self):
        """(Initialization function) Create a `TaskScheduler` instance.
        """
        self._task_sched = None

        # Parse scheduler type from configuration.
        sched_type = self._config.sched_type
        if sched_type == 'fifo':
            # self._task_sched = FIFOTaskScheduler(
            #     self._pipelines, self._task_execs)
            self._task_sched = FIFOTaskScheduler(self._task_execs)
        else:
            raise ValueError(f'Invalid scheduler type - {sched_type}')

    # def _on_task_ready(self):
    #     """Contains `Engine`-specific code blocks that need to be executed when a task
    #     is marked as ready.
    #     """
    #     pass

    # def _on_task_start(self):
    #     """Contains `Engine`-specific code blocks that need to be executed when a task
    #     begins running.
    #     """
    #     pass

    # def _on_task_finish(self):
    #     """Contains `Engine`-specific code blocks that need to be executed when a task
    #     is finished executing.
    #     """
    #     pass


_engine_lock = threading.Lock()


def start_engine():
    engine = Engine()
    if engine.is_alive():
        return

    with _engine_lock:
        if engine.is_alive():
            return
        engine.start()
