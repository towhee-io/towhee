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
from typing import List
import random
import weakref
import threading

from towhee.engine.thread_pool_task_executor import ThreadPoolTaskExecutor


class TaskScheduler(ABC):
    """Task scheduler abstract interface.

    Args:
        task_execs: (`List[towhee.TaskExecutor]`)
            A list of task executors that the `Engine` manages. Also should be
            continuously changing as new executors are acquired.
    """

    def __init__(self, task_execs: List[ThreadPoolTaskExecutor]):
        self._task_execs = task_execs
        self._graph_ctx_refs = []
        self._need_stop = False
        self._event = threading.Event()

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def join(self) -> None:
        raise NotImplementedError

    def schedule_forever(self, sleep_ms):
        """Runs the a single schedule step in a loop.

        sleep_ms: (`int`)
            Milliseconds to sleep after completing a single scheduling step.
        """
        while not self._need_stop:
            self.schedule_step()
            self._event.wait(sleep_ms / 1000)

    @abstractmethod
    def schedule_step(self):
        raise NotImplementedError


class BasicScheduler(TaskScheduler):
    """
    Basic scheduler.

    This scheduler will start all ops, and has no scheduling logic, so the
    schedule_step does nothing.
    """

    def __init__(self, task_execs: List[ThreadPoolTaskExecutor], threshold: int):
        super().__init__(task_execs)
        self._lock = threading.Lock()
        self._df_threshold = threshold

    def register(self, graph_ctx):
        for op in graph_ctx.op_ctxs.values():
            executor = self._find_optimal_exec()
            op.start(executor)

        with self._lock:
            self._graph_ctx_refs.append(weakref.ref(graph_ctx))

    def stop(self):
        with self._lock:
            for g_ctx_ref in self._graph_ctx_refs:
                g_ctx = g_ctx_ref()
                if g_ctx is not None:
                    g_ctx.stop()
        self._event.set()

    def join(self):
        with self._lock:
            for g_ctx_ref in self._graph_ctx_refs:
                g_ctx = g_ctx_ref()
                if g_ctx is not None:
                    g_ctx.join()

    def _remove_finished_graph(self):
        start_index = 0
        with self._lock:
            for g_ctx_ref in self._graph_ctx_refs:
                if g_ctx_ref() is not None:
                    break
                start_index += 1
            if start_index != 0:
                self._graph_ctx_refs = self._graph_ctx_refs[start_index:]

    def _scheduler_ops(self):
        # threshold <=0: disable scheduler
        if self._df_threshold <= 0:
            return

        for g_ctx_ref in self._graph_ctx_refs:
            g_ctx = g_ctx_ref()
            if g_ctx is not None:
                for name, df in g_ctx.dataframes.items():
                    if df.current_size > self._df_threshold:
                        g_ctx.slow_down(name, df.current_size / self._df_threshold)
                    elif df.current_size == 0 and not df.sealed:
                        g_ctx.speed_up(name)
                    else:
                        pass

    def _df_gc(self):
        for g_ctx_ref in self._graph_ctx_refs:
            g_ctx = g_ctx_ref()
            if g_ctx is not None:
                g_ctx.gc()

    def schedule_step(self):
        """
        Scheduler:
            1. Remove the graph which is finished.
            2. Call df gc.

        """
        self._remove_finished_graph()
        self._df_gc()
        self._scheduler_ops()

    def _find_optimal_exec(self):
        """
        Acquires the least busy instance of `TaskExecutor` that can still execute the operator.
        """
        return random.choice(self._task_execs)
