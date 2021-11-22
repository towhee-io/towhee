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

from typing import Tuple, Dict


from towhee.engine.operator_runner.map_runner import MapRunner


class FilterRunner(MapRunner):
    """
    FilterRunner

    Filter operator's output must be a bool type, if the output is true, pass data to
    the next operator else drop it.
    """

    def _get_inputs(self) -> Tuple[bool, Dict[str, any]]:
        try:
            data, self._row_data = self._reader.read()
            return False, data
        except StopIteration:
            return True, None

    def _set_outputs(self, output: bool):
        if not isinstance(output, bool):
            raise RuntimeError('Filter operator\'s output must be a bool type, this one is {}'.format(type(output)))

        if output:
            self._writer.write(self._row_data)
