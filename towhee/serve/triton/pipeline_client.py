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
import json
import numpy as np

from towhee.utils.np_format import NumpyArrayDecoder, NumpyArrayEncoder
from towhee.serve.triton.constant import PIPELINE_NAME


class Client:
    """
    Triton HttpClient

    Args:
        url(`string`): IP address and port for the triton server, such as '<host>:<port>' and  '127.0.0.1:8001'.
        model_name(`string`): Model name in triton, defaults to 'pipeline'.

    Examples:
        >>> from towhee import triton_client
        >>> client = triton_client('<your-ip>:<your-port>')
        >>> res = client('your-data')
    """
    def __init__(self, url: str, model_name: str = PIPELINE_NAME):
        from towhee.utils.tritonclient_utils import httpclient  # pylint: disable=import-outside-toplevel
        self._client = httpclient.InferenceServerClient(url)
        self._model_name = model_name

    def __call__(self, *args):
        inputs = self._solve_inputs(args)
        response = self._client.infer(self._model_name, inputs)
        outputs = self._solve_responses(response)
        return outputs

    def batch(self, args_list):
        inputs = self._solve_inputs(args_list, len(args_list))
        response = self._client.infer(self._model_name, inputs)
        outputs = self._solve_responses(response)
        return outputs

    def close(self):
        self._client.close()

    @staticmethod
    def _solve_inputs(args, size=1):
        from towhee.utils.tritonclient_utils import httpclient  # pylint: disable=import-outside-toplevel
        inputs = [httpclient.InferInput('INPUT0', [size, 1], 'BYTES')]
        arg_list = []
        for arg in args:
            arg_list.append([json.dumps(arg, cls=NumpyArrayEncoder)])
        data = np.array(arg_list, dtype=np.object_)
        inputs[0].set_data_from_numpy(data)
        return inputs

    @staticmethod
    def _solve_responses(response):
        res = response.as_numpy('OUTPUT0')[0]
        outputs = json.loads(res, cls=NumpyArrayDecoder)
        return outputs
