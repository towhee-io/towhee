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

from typing import Dict
from abc import ABC, abstractmethod
from enum import Enum, auto
from threading import Event

from towhee.utils.log import engine_log


class RunnerStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()


class _OpInfo:
    """
    Operator hub info.
    """

    def __init__(self, op_name: str, hub_op_id: str,
                 op_args: Dict[str, any]) -> None:
        self._op_name = op_name
        self._hub_op_id = hub_op_id
        self._msg = None
        self._op_args = op_args
        self._op_key = _OpInfo._calc_op_key(hub_op_id, op_args)

    @property
    def op_key(self):
        return self._op_key

    @property
    def op_name(self):
        return self._op_name

    @property
    def hub_op_id(self):
        return self._hub_op_id

    @property
    def op_args(self):
        return self._op_args

    @staticmethod
    def _calc_op_key(hub_op_id: str, op_args: Dict[str, any]):
        if op_args:
            args_tup = tuple(sorted(op_args.items()))
        else:
            args_tup = ()
        return (hub_op_id, ) + args_tup


class RunnerBase(ABC):
    """
    Running instance of op.

    The base class provides some function to control status.
    """

    def __init__(self, name: str, index: int,
                 op_name: str, hub_op_id: str,
                 op_args: Dict[str, any],
                 reader=None, writer=None) -> None:
        self._name = name
        self._index = index
        self._status = RunnerStatus.IDLE
        self._op_info = _OpInfo(op_name, hub_op_id, op_args)
        self._reader = reader
        self._writer = writer
        self._need_stop = False
        self._end_event = Event()

    @property
    def op_key(self):
        return self._op_info.op_key

    def _set_end_status(self, status: RunnerStatus):
        self._set_status(status)
        self._end_event.set()
        engine_log.info('%s ends with status: %s', self._name, status)

    @property
    def op_name(self):
        return self._op_info.op_name

    @property
    def hub_op_id(self):
        return self._op_info.hub_op_id

    @property
    def op_args(self):
        return self._op_info.op_args

    @property
    def status(self) -> RunnerStatus:
        return self._status

    @property
    def msg(self):
        return self._msg

    def _set_finished(self) -> None:
        self._set_end_status(RunnerStatus.FINISHED)

    def _set_idle(self) -> None:
        self._set_status(RunnerStatus.IDLE)

    def _set_failed(self, msg: str) -> None:
        error_info = '{} runs failed, error msg: {}'.format(str(self), msg)
        engine_log.error(error_info)
        self._msg = error_info
        self._set_end_status(RunnerStatus.FAILED)

    def _set_running(self) -> None:
        self._set_status(RunnerStatus.RUNNING)

    def _set_status(self, status: RunnerStatus) -> None:
        self._status = status

    def is_idle(self) -> bool:
        return self.status == RunnerStatus.IDLE

    def is_finished(self) -> bool:
        return self.status == RunnerStatus.FINISHED

    def set_stop(self) -> None:
        engine_log.info('Begin to stop %s', str(self))
        self._need_stop = True
        self._reader.close()

    def __str__(self) -> str:
        return '{}:{}'.format(self._name, self._index)

    @abstractmethod
    def process_step(self) -> bool:
        raise NotImplementedError

    def join(self):
        self._end_event.wait()

    def process(self):
        engine_log.info('Begin to run %s', str(self))
        self._set_running()
        while True:
            if not self._need_stop:
                if self.process_step():
                    break
            else:
                self._set_finished()
                break
