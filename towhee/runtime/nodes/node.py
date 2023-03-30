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


from typing import List
from enum import Enum, auto
from abc import ABC
import traceback

from towhee.runtime.data_queue import DataQueue
from towhee.runtime.runtime_conf import set_runtime_config
from towhee.runtime.constants import OPType
from towhee.runtime.time_profiler import Event, TimeProfiler
from towhee.utils.log import engine_log


class NodeStatus(Enum):
    NOT_RUNNING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()
    STOPPED = auto()

    @staticmethod
    def is_end(status: 'NodeStatus') -> bool:
        return status in [NodeStatus.FINISHED, NodeStatus.FAILED, NodeStatus.STOPPED]


class Node(ABC):
    """
    node_info:
        name
        func_type: operator/lambda
        node_type: map/filter/flat_map/window/time_window/window_all
        input_schema: (name, name)
        output_schema: (name, name, name)
        # operator meta info:
        op_info: {
            hub_id: '',
            name: '',
            args: [],
            kwargs: {},
            tag: ''
        },
        config: {}
    """
    def __init__(self, node_repr: 'NodeRepr',
                 op_pool: 'OperatorPool',
                 in_ques: List[DataQueue],
                 out_ques: List[DataQueue],
                 time_profiler: 'TimeProfiler'):

        self._node_repr = node_repr
        self._op_pool = op_pool
        if time_profiler is None:
            self._time_profiler = TimeProfiler()
        else:
            self._time_profiler = time_profiler
        self._op = None

        self._in_ques = in_ques
        self._output_ques = out_ques
        self._status = NodeStatus.NOT_RUNNING
        self._need_stop = False
        self._err_msg = None

    def initialize(self) -> bool:
        #TODO
        # Create multiple-operators to support parallelism.
        # Read the parallelism info by config.
        op_type = self._node_repr.op_info.type
        if op_type in [OPType.HUB, OPType.BUILTIN]:
            try:
                hub_id = self._node_repr.op_info.operator
                with set_runtime_config(self._node_repr.config):
                    self._time_profiler.record(self.uid, Event.init_in)
                    self._op = self._op_pool.acquire_op(
                        self.uid,
                        hub_id,
                        self._node_repr.op_info.init_args,
                        self._node_repr.op_info.init_kws,
                        self._node_repr.op_info.tag,
                        self._node_repr.op_info.latest,
                    )
                    self._time_profiler.record(self.uid, Event.init_out)
                    return True
            except Exception as e:  # pylint: disable=broad-except
                st_err = '{}, {}'.format(str(e), traceback.format_exc())
                err = 'Create {} operator {}:{} with args {} and kws {} failed, err: {}'.format(
                    self.name,
                    hub_id,
                    self._node_repr.op_info.tag,
                    str(self._node_repr.op_info.init_args),
                    str(self._node_repr.op_info.init_kws),
                    str(st_err))
                self._set_failed(err)
            return False
        elif op_type in [OPType.LAMBDA, OPType.CALLABLE]:
            self._op = self._node_repr.op_info.operator
            return True
        else:
            err = 'Unkown callable type {}'.format(op_type)
            self._set_failed(err)
            return False

    @property
    def name(self):
        # TODO
        # Can be named in config.
        return self._node_repr.name

    @property
    def uid(self):
        return self._node_repr.uid

    @property
    def status(self):
        return self._status

    @property
    def err_msg(self):
        return self._err_msg

    def _set_finished(self) -> None:
        self._set_status(NodeStatus.FINISHED)
        for out in self._output_ques:
            out.seal()

    def _set_end_status(self, status: NodeStatus):
        self._set_status(status)
        engine_log.info('%s ends with status: %s', self.name, status)
        for que in self._in_ques:
            que.seal()
        for out in self._output_ques:
            out.clear_and_seal()

    def _set_stopped(self) -> None:
        self._set_end_status(NodeStatus.STOPPED)

    def _set_failed(self, msg: str) -> None:
        error_info = '{} runs failed, error msg: {}'.format(str(self), msg)
        self._err_msg = error_info
        self._set_end_status(NodeStatus.FAILED)

    def _call(self, inputs):
        try:
            return True, self._op(*inputs), None
        except Exception as e:  # pylint: disable=broad-except
            err = '{}, {}'.format(str(e), traceback.format_exc())
            return False, None, err

    def process_step(self) -> bool:
        raise NotImplementedError

    def process(self):
        engine_log.info('Begin to run %s', str(self))
        self._set_status(NodeStatus.RUNNING)
        while not self._need_stop and not NodeStatus.is_end(self.status):
            try:
                self.process_step()
            except Exception as e:  # pylint: disable=broad-except
                err = '{}, {}'.format(e, traceback.format_exc())
                self._set_failed(err)

    def data_to_next(self, data) -> bool:
        for out_que in self._output_ques:
            if not out_que.put_dict(data):
                self._set_stopped()
                return False
            pass
        return True

    def _set_status(self, status: NodeStatus) -> None:
        self._status = status

    def __str__(self) -> str:
        return 'Node-{}'.format(self.name)

    def __del__(self):
        if self._node_repr.op_info.type == OPType.HUB and self._op:
            self._op_pool.release_op(self._op)
