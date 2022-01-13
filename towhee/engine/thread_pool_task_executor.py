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

import os
import signal
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from towhee.engine.operator_pool import OperatorPool
from towhee.engine.operator_runner.runner_base import RunnerBase
from towhee.utils.log import engine_log


class ThreadPoolTaskExecutor(threading.Thread):
    """
    A thread pool executor. Each device has one Executor.

    Args:
        name: (`str`)
            Name of the device for which the executor will run. Example device name:
                dev_name_0 = 'cpu:0'
                dev_name_1 = 'gpu:0'
        cache_path: (`str`)
            Local path for which operators are stored. Defaults to
            `$HOME/.towhee/operators`.
    """
    def __init__(self, name: str, cache_path: str = None):
        super().__init__()
        self._name = name
        self._task_queue = queue.Queue()
        self._op_pool = OperatorPool(cache_path=cache_path)
        self._thread_pool = ThreadPoolExecutor()
        self._is_run = True
        self.setDaemon(True)

    @property
    def name(self):
        return self._name

    @property
    def num_tasks(self):
        return self._task_queue.qsize()

    def push_task(self, task: RunnerBase) -> bool:
        """Push a task to the end of the task queue.

        Args:
            task:
                Pre-initialized `Task` object to push onto the queue.
        """
        return self._task_queue.put(task)

    def execute(self, runner: RunnerBase):
        try:
            op = self._op_pool.acquire_op(runner.hub_op_id, runner.op_args, runner.tag)
            runner.set_op(op)
            runner.process()
            if runner.is_end() and runner.op is not None:
                runner.unset_op()
                self._op_pool.release_op(op)
        except Exception as e:  # pylint: disable=broad-except
            engine_log.error(traceback.format_exc())
            engine_log.error(e)
            os.kill(os.getpid(), signal.SIGINT)

    def run(self):
        """
        Runs the execution loop.
        """

        while self._is_run:

            # If there are no tasks in the queue, will blocking until sth put into the queue.
            # When the executor ends, self.stop() function will put None into the queue,
            # so if get None, break the loop.
            runner = self._task_queue.get()

            if self._is_run and runner is not None:
                self._thread_pool.submit(self.execute, runner)
            else:
                break

    def stop(self):
        """
        Sets a flag, which stops the execution loop after a period of time.
        """
        self._is_run = False
        self._task_queue.put(None)
