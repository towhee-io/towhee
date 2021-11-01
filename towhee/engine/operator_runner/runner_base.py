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
from enum import Enum

from towhee.utils.log import engine_log


class RunnerStatus(Enum):
    IDLE = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3
    STOPPED = 4


class RunnerBase(ABC):
    def __init__(self, name: str, index: int):
        self._name = name
        self._index = index
        self._status = RunnerStatus.IDLE
        self._need_stop = False

    @property
    def status(self) -> RunnerStatus:
        return self._status

    def _set_finished(self) -> None:
        self._set_status(RunnerStatus.FINISHED)

    def _set_idle(self) -> None:
        self._set_status(RunnerStatus.IDLE)

    def _set_failed(self, msg: str) -> None:
        engine_log.error('{} failed'.format(str(self)))
        self._msg = msg
        self._set_status(RunnerStatus.FAILED)

    def _set_running(self) -> None:
        self._set_status(RunnerStatus.RUNNING)
        

    def _set_status(self, status: RunnerStatus) -> None:
        self._status = status

    def is_idle(self) -> bool:
        return self.status == RunnerStatus.IDLE

    def is_stop(self) -> bool:
        return self.status == RunnerStatus.STOPPED

    def set_stop(self) -> None:
        self._need_stop = True

    def __str__(self) -> str:
        return '{}:{}'.format(self._name, self._index)

    @abstractmethod
    def process_step(self) -> bool:
        raise NotImplementedError

    def process(self):
        engine_log.info('Begin run {}'.format(str(self)))
        self._set_running()
        while not self._need_stop:
            if self.process_step():
                break

