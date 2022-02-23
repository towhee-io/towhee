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

from towhee.engine.operator_runner.runner_base import RunnerBase
from towhee.errors import OpIOTypeError


class FilterRunner(RunnerBase):
    """
    FilterRunner

    Filter operator's output must be a bool type, if the output is true, pass data to
    the next operator else drop it.
    """

    def _set_outputs(self, output: bool):
        if not isinstance(output, bool):
            raise OpIOTypeError('Filter operator\'s output must be a bool type, this one is {}'.format(type(output)))

        if output:
            self._writer.write(self._row_data)
        else:
            self._frame.empty = True
            self._writer.write((self._frame, ))
