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
import time
from typing import Dict, Tuple
from abc import ABC
from enum import Enum, auto
from threading import Event
import traceback

from towhee.types._frame import FRAME, _Frame
from towhee.engine.status import Status
from towhee.utils.log import engine_log
from towhee.hparam import param_scope


class RunnerStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()

    @staticmethod
    def is_end(status: 'RunnerStatus') -> bool:
        return status in [RunnerStatus.FINISHED, RunnerStatus.FAILED]


class _OpInfo:
    """
    Operator hub info.
    """

    def __init__(self, op_name: str, hub_op_id: str, op_args: Dict[str, any], tag: str) -> None:
        self._op_name = op_name
        self._hub_op_id = hub_op_id
        self._tag = tag
        self._op_args = op_args

    @property
    def op_name(self):
        return self._op_name

    @property
    def hub_op_id(self):
        return self._hub_op_id

    @property
    def op_args(self):
        return self._op_args

    @property
    def tag(self):
        return self._tag


class RunnerBase(ABC):
    """
    Running instance of op.

    The base class provides some function to control status.
    """

    def __init__(
        self,
        name: str,
        index: int,
        op_name: str,
        tag: str,
        hub_op_id: str,
        op_args: Dict[str, any],
        readers=None,
        writer=None,
    ) -> None:
        self._name = name
        self._index = index
        self._status = RunnerStatus.IDLE
        self._msg = None
        self._op_info = _OpInfo(op_name, hub_op_id, op_args, tag)

        self._readers = readers
        self._sleep_time = 0

        # only concat can have multiple readers
        if self._readers:
            self._reader = readers[0]
        self._writer = writer
        self._need_stop = False
        self._end_event = Event()
        self._op = None

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
    def tag(self):
        return self._op_info.tag

    @property
    def op_args(self):
        return self._op_info.op_args

    @property
    def status(self) -> RunnerStatus:
        return self._status

    @property
    def msg(self):
        return self._msg

    @property
    def op(self):
        return self._op

    def is_end(self) -> bool:
        return RunnerStatus.is_end(self.status)

    def set_op(self, op):
        self._op = op

    def unset_op(self):
        self._op = None

    def _set_finished(self) -> None:
        self._set_end_status(RunnerStatus.FINISHED)

    def _set_idle(self) -> None:
        self._set_status(RunnerStatus.IDLE)

    def _set_failed(self, msg: str) -> None:
        error_info = '{} runs failed, error msg: {}'.format(str(self), msg)
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
        for reader in self._readers:
            reader.close()

    def __str__(self) -> str:
        return '{}:{}'.format(self._name, self._index)

    def _call_op(self, inputs) -> Status:
        try:
            with param_scope() as hp:
                hp().towhee.need_schema = True
                outputs = self._op(**inputs)
            return Status.ok_status(outputs)
        except Exception as e:  # pylint: disable=broad-except
            err = '{}, {}'.format(str(e), traceback.format_exc())
            return Status.err_status(err)

    def _get_inputs(self) -> Tuple[bool, Dict[str, any]]:
        try:
            data, self._row_data = self._reader.read()
            assert isinstance(self._row_data[-1], _Frame)
            self._frame = self._row_data[-1]
            self._frame.prev_id = self._frame.row_id
            return False, data
        except StopIteration:
            return True, None

    def _set_outputs(self, output: any):
        if hasattr(output, '_asdict'):
            output = output._asdict()

        if isinstance(output, dict):
            output.update({FRAME: self._frame})

        self._writer.write(output)

    def process_step(self) -> bool:
        is_end, op_input_params = self._get_inputs()
        if is_end:
            self._set_finished()
            return True

        if op_input_params is None:
            # No data in dataframe, but dataframe is not sealed
            self._set_idle()
            return True

        st = self._call_op(op_input_params)
        if st.is_ok():
            self._set_outputs(st.data)
            return False
        else:
            self._set_failed(st.msg)
            return True

    def slow_down(self, time_sec: float):
        self._sleep_time = time_sec

    def speed_up(self):
        self._sleep_time = 0

    def sleep(self):
        if self._sleep_time > 0:
            time.sleep(self._sleep_time)

    def join(self):
        self._end_event.wait()

    def process(self):
        engine_log.info('Begin to run %s', str(self))
        self._set_running()
        while True:
            if not self._need_stop:
                try:
                    if self.process_step():
                        break
                except Exception as e:  # pylint: disable=broad-except
                    err = '{}, {}'.format(e, traceback.format_exc())
                    self._set_failed(err)
                    break
            else:
                self._set_finished()
                break
            self.sleep()
