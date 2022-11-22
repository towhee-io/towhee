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


from .node import Node
from towhee.runtime.performance_profiler import Event


class Output(Node):
    """Output the data as input

       Examples:
           p1 = towhee.pipe.input('url').output('url')
    """
    def process_step(self) -> bool:
        self._time_profiler.record(self.uid, Event.queue_in)
        all_data = {}
        for q in self._in_ques:
            data = q.get_dict()
            if data:
                all_data.update(data)

        if not all_data:
            self._set_finished()
            return True

        self._time_profiler.record(self.uid, Event.process_in)
        self._time_profiler.record(self.uid, Event.process_out)

        for out_que in self._output_ques:
            if not out_que.put_dict(all_data):
                self._set_stopped()
                return True
        self._time_profiler.record(self.uid, Event.queue_out)
        return False
