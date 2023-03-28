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


class Event:
    pipe_name = '_run_pipe'
    pipe_in = 'pipe_in'
    pipe_out = 'pipe_out'
    init_in = 'init_in'
    init_out = 'init_out'
    process_in = 'process_in'
    process_out = 'process_out'
    queue_in = 'queue_in'
    queue_out = 'queue_out'


class TimeProfiler:
    """
    TimeProfiler to record the event and timestamp.
    """
    def __init__(self, enable=False, time_record=None):
        self._enable = enable
        self._time_record = time_record if time_record else []
        self.inputs = None

    def record(self, uid, event):
        if not self._enable:
            return
        timestamp = int(round(time.time() * 1000000))
        self._time_record.append(f'{uid}::{event}::{timestamp}')

    @property
    def time_record(self):
        return self._time_record

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def reset(self):
        self._time_record = []
