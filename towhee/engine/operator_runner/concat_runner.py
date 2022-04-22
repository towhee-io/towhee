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

from towhee.engine.operator_runner.runner_base import RunnerBase, OpInfo


class ConcatRunner(RunnerBase):
    """
    Concat wrapper, multiple inputs one output.

    Concat can not run in multi-thread.
    """

    def __init__(
        self,
        op_info: OpInfo,
        name: str,
        index: int,
        readers=None,
        writer=None,
    ) -> None:
        if len(readers) <= 1:
            self._set_failed('Concat operator\'s inputs are at leaset two, current is %s' % len(readers))
            return

        super().__init__(op_info, name, index, readers, writer)

    def process_step(self):
        self._op._readers = self._readers  # pylint: disable=protected-access
        self._op._writer = self._writer  # pylint: disable=protected-access
        self._op()
        self._set_finished()
        return True
