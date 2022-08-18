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


import logging
from pathlib import Path

from towhee import ops
from towhee.serve.triton.bls.model_runner.transformer import RequestToOpInputs, OpOutputToResponses
from towhee.serve.triton import constant
from towhee.serve.triton.bls.utils.op_config import OpConfig

logger = logging.getLogger()


class TritonPythonModel:
    '''
    Supports running towhee map operators.
    '''
    def initialize(self, args):
        device = None
        if args['model_instance_kind'] == 'GPU':
            device = int(args['model_instance_device_id'])

        self._op_config = OpConfig.load_from_file(self.op_config_file)
        if self._op_config is None:
            err = 'Load operator config file [%s] failed' % self.op_config_file
            raise IOError(err)

        hub = getattr(ops, self._op_config.op_hub)
        op_wrapper = getattr(hub, self._op_config.op_name)(**self._op_config.init_args)
        self._op = op_wrapper.get_op()
        if hasattr(self._op, 'to_device') and device is not None:
            self._op._device = device  # pylint: disable=protected-access
            self._op.to_device()

    @property
    def op_config_file(self) -> str:
        if hasattr(self, '_op_config_file'):
            return self._op_config_file
        return str(Path(__file__).parent.resolve() / constant.OP_CONFIG_FILE)

    def execute(self, requests):
        responses = []
        for request in requests:
            inputs = RequestToOpInputs(request, self._op.input_schema()).get_towhee_data()
            ret = self._op(*inputs)
            if not isinstance(ret, tuple):
                ret = (ret, )
            response = OpOutputToResponses(ret).to_triton_responses()
            responses.append(response)
        return responses

    def finalize(self):
        pass
