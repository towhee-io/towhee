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
from typing import List
from collections import deque

from towhee.runtime.time_profiler import Event
from towhee.runtime.data_queue import Empty

from .node import Node
from ._single_input import SingleInputMixin


class Reduce(Node, SingleInputMixin):
    """
    Reduce node.

    Reduce the sequence to a single value

    """

    def __init__(self, node_repr: 'NodeRepr',
                 op_pool: 'OperatorPool',
                 in_ques: List['DataQueue'],
                 out_ques: List['DataQueue'],
                 time_profiler: 'TimeProfiler'):

        super().__init__(node_repr, op_pool, in_ques, out_ques, time_profiler)
        self._col_cache = dict((key, deque()) for key in self._node_repr.inputs)
        self._lock = threading.Lock()

    def _read_from_dq(self):
        data = self.input_que.get_dict()
        if data is None:
            return True

        if not self.side_by_to_next(data):
            return False

        for k, v in self._col_cache.items():
            if data[k] is not Empty():
                v.append(data[k])
        return True

    def get_col(self, key: str):
        while True:
            if len(self._col_cache[key]) != 0:
                yield self._col_cache[key].popleft()
                continue

            with self._lock:
                if len(self._col_cache[key]) != 0:
                    yield self._col_cache[key].popleft()
                    continue

                if self._read_from_dq() and len(self._col_cache[key]) != 0:
                    yield self._col_cache[key].popleft()
                    continue
            break

    def process_step(self):
        self._time_profiler.record(self.uid, Event.queue_in)
        self._time_profiler.record(self.uid, Event.process_out)
        succ, outputs, msg = self._call([self.get_col(key) for key in self._node_repr.inputs])
        assert succ, msg
        self._time_profiler.record(self.uid, Event.process_out)
        size = len(self._node_repr.outputs)
        if size > 1:
            output_map = dict((self._node_repr.outputs[i], outputs[i])
                              for i in range(size))
        elif size == 0:
            # ignore the op result
            # eg: ignore the milvus result
            # .map('vec', (), ops.ann_insert.milvus()),
            output_map = {}
        else:
            # Use one col to store all op result.
            output_map = {}
            output_map[self._node_repr.outputs[0]] = outputs

        self._time_profiler.record(self.uid, Event.queue_out)
        if not self.data_to_next(output_map):
            return
        self._set_finished()
