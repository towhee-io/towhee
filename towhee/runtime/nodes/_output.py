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
from ._single_input import SingleInputMixin
from towhee.runtime.time_profiler import Event


class Output(Node, SingleInputMixin):
    """Output the data as input

       Examples:
           p1 = towhee.pipe.input('url').output('url')
    """
    def initialize(self):
        self._output_ques[0].max_size = 0
        return super().initialize()

    def process_step(self):
        self._time_profiler.record(self.uid, Event.queue_in)

        data = self.read_row()
        if data is None:
            return

        self._time_profiler.record(self.uid, Event.process_in)
        self._time_profiler.record(self.uid, Event.process_out)

        if not self.data_to_next(data):
            return

        self._time_profiler.record(self.uid, Event.queue_out)

