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

from typing import Generator
from copy import deepcopy


from towhee.engine.operator_runner.runner_base import RunnerBase
from towhee.types._frame import FRAME
from towhee.errors import OpIOTypeError


class GeneratorRunner(RunnerBase):
    """
    GeneratorRunner, the ops must return a generator.

    """

    def _set_outputs(self, output: Generator):
        if not isinstance(output, Generator):
            raise OpIOTypeError("Op {}'s output is not a generator".format(self.op_name))

        for data in output:
            frame = deepcopy(self._frame)
            if frame.parent_path == '':
                frame.parent_path = str(frame.prev_id)
            else:
                frame.parent_path = '-'.join([frame.parent_path, str(frame.prev_id)])

            item = data._asdict()

            if item.get('TIMESTAMP') is not None:
                frame.timestamp = item['TIMESTAMP']
                del item['TIMESTAMP']
            item.update({FRAME: frame})
            self._writer.write(item)
            self.sleep()
