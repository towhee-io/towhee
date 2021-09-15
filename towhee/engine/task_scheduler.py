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
from collections import defaultdict
import random
import time
from typing import List

from towhee.engine.pipeline import Pipeline
from towhee.engine.task import Task
from towhee.engine.task_executor import TaskExecutor


MAX_EXECUTOR_TASKS = 10


class _TaskScheduler(ABC):
    """Task scheduler abstract interface.

    Args:
        executors: (`List[towhee.TaskExecutor]`)
            A list of task executors that the `Engine` manages. Also should be
            continuously changing as new executors are acquired.
    """

    def __init__(self, task_execs: List[TaskExecutor]):
        self._task_execs = task_execs
        self._pipelines = []

    def add_pipeline(self, pipeline: Pipeline):
        """Add a single pipeline for this scheduler to manage tasks for.

        Args:
            pipeline: `towhee.Pipeline`
                A single pipeline to schedule tasks for.
        """
        pipeline.add_task_ready_handler(self._on_task_ready)
        pipeline.add_task_start_handler(self._on_task_start)
        pipeline.add_task_finish_handler(self._on_task_finish)
        self._pipelines.append(pipeline)

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

    Args:
        task_execs: (`List[towhee.TaskExecutor]`)
            See `_TaskScheduler` docstring.
        sleep_ms: (`int`)
            Milliseconds to sleep for after looping through all operator contexts. The
            `TaskExecutor` instances will maintain greater activity with a lower value,
            but the scheduler will be more inclined to prioritize early
    """

    def __init__(self, task_execs: List[TaskExecutor], sleep_ms: int = 1):
        super().__init__(task_execs)
        self._sleep_ms = sleep_ms

        # `FIFOTaskScheduler` maintains a dictionary of `hub_op_id` to
        # `List[TaskExecutor]` mappings, tracking which operator IDs are loaded into
        # which task executors.
        self._op_id_exec_map = defaultdict(list)

    def execute(self):
        """This function loops continuously through all operator contexts in all graphs,
        adding a single task from individual operator contexts as appropriate. We do
        this continuously until one or more `TaskExecutor` instances are full, then wait
        a bit before retrying. For maximum efficiency, we prioritize operators towards
        the end of the graph first, greedily prioritizing dataframe outputs.
        """

        while True:
            for pipeline in self._pipelines:
                for op_ctx in pipeline.graph_ctx.op_ctxs[::-1]:
                    tasks = op_ctx.pop_ready_tasks(n=1)
                    if not tasks:
                        continue
                    task = tasks[0]

                    # If `push_task` returns `False`, then the optimal executor is
                    # already full, wait a while and try again.
                    task_exec = self._find_optimal_exec(task)
                    while not task_exec.push_task(task):
                        time.sleep(0.001)
                        task_exec = self._find_optimal_exec(task)

            time.sleep(self._sleep_ms)

    def _find_optimal_exec(self, task: Task):
        """Acquires the least busy instance of `TaskExecutor` that can still execute
        the operator.

        Args:
            task: (`towhee.Task`)
                The `Task` instance that the executor will need to complete.
        """

        # Attempt to find the least busy executor with the model already loaded. We do
        # not consider executors that have more than `MAX_EXECUTOR_TASKS` queued up.
        min_num_tasks = MAX_EXECUTOR_TASKS
        optimal_exec = None
        for task_exec in self._op_id_exec_map[task.op_key]:
            if task_exec.num_tasks < min_num_tasks:
                optimal_exec = task_exec
                min_num_tasks = task_exec.num_tasks

        # TODO(fzliu): If no task executor with the specified object was found, assign
        # the least busy executor that has the resource (i.e. GPU, TPU, FPGA, etc...)
        # required by the task.
        if not optimal_exec:
            optimal_exec = random.choice(self._task_execs)
            self._op_id_exec_map[task.op_key].append(optimal_exec)

        return optimal_exec

    def on_task_ready(self):
        pass

    def on_task_start(self):
        pass

    def on_task_finish(self):
        pass
