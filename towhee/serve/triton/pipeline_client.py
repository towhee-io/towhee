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

import asyncio
from typing import List

import numpy as np
from towhee.serve.triton.serializer import to_triton_data, from_triton_data
from towhee.serve.triton.constant import PIPELINE_NAME
from towhee.utils.log import engine_log

from towhee.utils.triton_httpclient import aio_httpclient


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
        >>> client.close()

        You can also run as following:
        >>> with triton_client('<your-ip>:<your-port>') as client:
        ...     res = client('your-data')

        Batch interface.
        >>> data = [data1, data2, data3, data4, data5]
        >>> res = client(data, batch_size=4, safe=True)
        >>> client.close()

        You can also run as following:
        >>> with triton_client('<your-ip>:<your-port>') as client:
        ...     res = client(data, batch_size=4)
    """

    def __init__(self, url: str, model_name: str = PIPELINE_NAME):
        self._loop = asyncio.get_event_loop()
        self._client = aio_httpclient.InferenceServerClient(url)
        self._model_name = model_name

    async def _safe_call(self, inputs):
        try:
            response = await self._client.infer(self._model_name, inputs)
        except Exception as e:  # pylint: disable=broad-except
            engine_log.error(str(e))
            return [None] * inputs[0].shape()[0]
        return from_triton_data(response.as_numpy('OUTPUT0')[0])

    async def _call(self, inputs):
        response = await self._client.infer(self._model_name, inputs)
        return from_triton_data(response.as_numpy('OUTPUT0')[0])

    async def _multi_call(self, caller, batch_inputs):
        fs = []
        for inputs in batch_inputs:
            fs.append(caller(inputs))
        return await asyncio.gather(*fs)

    def __call__(self, *inputs):
        if len(inputs) > 1:
            inputs = [inputs]
        inputs = self._solve_inputs(inputs)
        return self._loop.run_until_complete(self._call(inputs))

    def batch(self, pipe_inputs: List, batch_size=4, safe=False):
        batch_inputs = [self._solve_inputs(pipe_inputs[i: i + batch_size]) for i in range(0, len(pipe_inputs), batch_size)]
        caller = self._safe_call if safe else self._call
        outputs = self._loop.run_until_complete(self._multi_call(caller, batch_inputs))
        ret = []
        for batch in outputs:
            for single in batch:
                ret.append(single)
        return ret

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
        self.close()

    def close(self):
        self._loop.run_until_complete(self._client.close())

    @staticmethod
    def _solve_inputs(pipe_inputs):
        batch_size = len(pipe_inputs)
        inputs = [aio_httpclient.InferInput('INPUT0', [batch_size, 1], 'BYTES')]
        batch = []
        for item in pipe_inputs:
            batch.append([to_triton_data(item)])
        data = np.array(batch, dtype=np.object_)
        inputs[0].set_data_from_numpy(data)
        return inputs

    def __del__(self):
        self.close()
