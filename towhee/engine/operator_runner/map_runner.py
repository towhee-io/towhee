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

from typing import Tuple, Union, Dict


from towhee.engine.operator_runner.runner_base import RunnerBase
from towhee.engine.status import Status
from towhee.utils.log import engine_log


class MapRunner(RunnerBase):
    """
    Map wrapper, one input one output.

    Wrapper will run in task executor in another thread or process.
    If run an op error, we should pass error info by an error handler.
    """

    def __init__(self, name: str, index: int,
                 op_name: str, hub_op_id: str,
                 op_args: Dict[str, any], reader, writer) -> None:
        super().__init__(name, index, op_name, hub_op_id, op_args, reader, writer)
        self._op = None

    def set_op(self, op):
        self._op = op

    def unset_op(self):
        raise NotImplementedError

    def _get_inputs(self) -> Tuple[bool, Dict[str, any]]:
        try:
            data = self._reader.read()
            if data is None:
                return False, None
            else:
                return False, data
        except StopIteration:
            return True, None


    def _set_outputs(self, output: any):
        self._writer.write(output)

    def _call_op(self, inputs) -> Tuple[bool, Union[str, any]]:
        try:
            outputs = self._op(**inputs)
            return Status.ok_status(outputs)
        except Exception as e:  # pylint: disable=broad-except
            engine_log.error(e)
            return Status.err_status(str(e))

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
