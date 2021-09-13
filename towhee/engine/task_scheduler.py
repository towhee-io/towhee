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
import time

from towhee.engine.task import Task


class _TaskScheduler(ABC):
    """Task scheduler abstract interface.

    Args:
        pipelines: (`List[towhee.Pipeline]`)
            A list of pipelines for which to acquire and schedule tasks. This list
            should be continuously changing as the engine receives new pipelines.
        executors: (`List[towhee.TaskExecutor]`)
            A list of task executors that the `Engine` manages. Also should be
            continuously changing as new executors are acquired.
    """

    def __init__(self, pipelines: List[Pipeline], task_execs: List[TaskExecutor]):
        self._pipelines = pipelines
        self._task_execs = task_execs

    @abstractmethod
    def execute(self):
        raise NotImplementedError

    @abstractmethod
    def on_task_ready(self):
        raise NotImplementedError

    @abstractmethod
    def on_task_start(self):
        raise NotImplementedError

    @abstractmethod
    def on_task_finish(self):
        raise NotImplementedError


class FIFOTaskScheduler(_TaskScheduler):
    """Very basic scheduler that sends tasks to executors in a first-in, first-out
    manner. The scheduler loops through all `OperatorContext` instances within the
    engine, acquiring tasks one at a time if available.
    """

    def execute(self):
        """This function loop continuously through all operator contexts in all graphs,
        adding a single task from individual operator contexts as appropriate. We do
        this continuously until one or more `TaskExecutor` instances are full, then wait
        a bit before retrying. For maximum efficiency, we prioritize operators towards
        the end of the graph first, greedily producing results.
        """

        while True:
            for pipeline in self._pipelines:
                for op_ctx in pipeline.graph_ctx.op_ctxs[::-1]:
                    tasks = op_ctx.pop_ready_tasks(n=1)
                    if not tasks:
                        continue

                    # TODO(fzliu): select an appropriate executor, given a "task
                    # requires", e.g. task.requires = 'gpu'.
                    task_exec = self._get_least_busy_task_exec()

                    # If `push_task` returns `False`, then the least busy executor
                    # is already full, wait a while and try again.
                    if not task_exec.push_task(tasks[0]):
                        time.sleep(1)

            time.sleep(0.001)

    def _get_least_busy_task_exec(self, requires: str = None):
        """Acquires the least busy valid instance of `TaskExecutor`.

        Args:
            requires: (`str`)
                A string denoting the type of resource to filter by. Executors which do
                not match the resource will not be considered by this function.
        """
        min_num_tasks = float('inf')
        least_busy_exec = None
        for task_exec in self._task_execs:
            if task_exec.num_tasks < min_num_tasks:
                least_busy_exec = task_exec
                min_num_tasks = task_exec.num_tasks
        return least_busy_exec

    def on_task_ready(self):
        pass

    def on_task_start(self):
        pass

    def on_task_finish(self):
        pass
