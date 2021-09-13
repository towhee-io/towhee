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


from typing import Any, Dict, List

# from towhee.engine.task import Task
from towhee.engine.scheduler import FIFOTaskScheduler
from towhee.engine.pipeline import Pipeline


class Engine:
    """Engines are the core component responsible for deliving results to the user. A
    single engine may be composed of multiple pipelines.

    Args:
        config: (`dict`)
            Engine configuration. Example configuration:
                config = {
                    'sched_type': 'fifo'
                }
        cache_path: (`str`)
            Local path for which operators are stored. Defaults to
            `$HOME/.towhee/cache`.
    """

    def __init__(self, config: Dict[str, Any], cache_path: str):
        self._config = config
        self._pipelines = []

        # Setup executors and scheduler.
        self._setup_task_execs()
        self._setup_task_sched()

    @property
    def pipelines(self):
        return self._pipelines

    @property
    def task_execs(self):
        return self._task_execs

    def add_pipeline(self, pipeline: Pipeline):
        """Add a single pipeline to this engine. Pipelines can be added long after an
        engine has been instantiated.

        Args:
            pipeline: `towhee.Pipeline`
                A single pipeline with which this engine will be assume execution
                responsibility.
        """
        pipeline.add_task_ready_handler(self._on_task_ready)
        pipeline.add_task_start_handler(self._on_task_start)
        pipeline.add_task_finish_handler(self._on_task_finish)
        self._pipelines.append(pipeline)

    def _setup_task_execs(self):
        """(Initialization function) Scan for devices and create TaskExecutors to
        manage task execution on CPU, GPU, and other devices.
        """
        self._task_execs = []

        # TODO(fzliu): Perform device scan.
        dev_names = ['cpu:0']

        # Create executor threads and begin running.
        for name in dev_names:
            executor = TaskExecutor(name=name)
            executor.start()
            self._task_execs.append(executor)

    def _setup_task_sched(self):
        """(Initialization function) Create a `TaskScheduler` instance.
        """
        self._task_sched = None

        # Parse scheduler type from configuration.
        sched_type = self._config.get('sched_type', 'fifo')
        if sched_type == 'fifo':
            self._task_sched = FIFOTaskScheduler(self._pipelines)
        else:
            raise ValueError('Invalid scheduler type - {0}'.format(sched_type))

    def _on_task_ready(self):
        """Contains `Engine`-specific code blocks that need to be executed when a task
        is marked as ready.
        """
        pass

    def _on_task_start(self):
        """Contains `Engine`-specific code blocks that need to be executed when a task
        begins running.
        """
        pass

    def _on_task_finish(self):
        """Contains `Engine`-specific code blocks that need to be executed when a task
        is finished executing.
        """
        pass
