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
import queue
import numpy as np
from functools import partial
from typing import Iterable


from towhee.utils.np_format import NumpyArrayDecoder, NumpyArrayEncoder
from towhee.serve.triton.constant import PIPELINE_NAME


class Client:
    """
    Triton http Client

    Args:
        url(`string`): IP address and HTTPService port for the triton server, such as '<host>:<port>' and  '127.0.0.1:8001'.
        model_name(`string`): Model name in triton, defaults to 'pipeline'.

    Examples:
        >>> from towhee import triton_client
        >>> client = triton_client('<your-ip>:<https-port>')
        >>> res = client('your-data')
        >>> client.close()

        You can also run as following:
        >>> with triton_client('<your-ip>:<your-port>') as client:
        ...     res = client('your-data')

        And run with bacth:
        >>> with triton_client('<your-ip>:<your-port>') as client:
        ...     res = client.batch(['your-data0', 'your-data1', 'your-data2'])
    """
    def __init__(self, url: str, model_name: str = PIPELINE_NAME):
        from towhee.utils.tritonclient_utils import httpclient  # pylint: disable=import-outside-toplevel
        self._client = httpclient.InferenceServerClient(url)
        self._model_name = model_name

    def __call__(self, *args):
        if len(args) > 1:
            args = [args]
        inputs = self._solve_inputs(args)
        response = self._client.infer(self._model_name, inputs)
        outputs = self._solve_responses(response)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
        self.close()

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


class UserData:
    def __init__(self):
        self.completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data.completed_requests.put((result, error))


class StreamClient:
    """
    Triton grpc StreamClient

    Args:
        url(`string`): IP address and GRPCInferenceService port for the triton server, such as '<host>:<port>' and  '127.0.0.1:8001'.
        model_name(`string`): Model name in triton, defaults to 'pipeline'.

    Examples:
        >>> from towhee import triton_stream_client
        >>> data = ['your-data0', 'your-data1', 'your-data2']
        >>> client = triton_stream_client('<your-ip>:<grpc-port>')
        >>> res = client(iter(data))
        >>> client.stop_stream()

        You can also run with batch as following:
        >>> data = [['your-data0', 'your-data1', 'your-data2']]
        >>> with triton_stream_client('<your-ip>:<grpc-port>') as client:
        ...     res = client(iter(data), batch_size=3)
    """
    def __init__(self, url: str, model_name: str = PIPELINE_NAME):
        from towhee.utils.tritonclient_utils import grpcclient  # pylint: disable=import-outside-toplevel
        self._client = grpcclient.InferenceServerClient(url)
        self._model_name = model_name
        self.user_data = UserData()
        self._client.start_stream(callback=partial(completion_callback, self.user_data))

    def __call__(self, iterator: Iterable, batch_size: int = 1):
        """
        Run with GRPC streamingAPI.

        Args:
            iterator(`Iterable`): The data to run.
            batch_size(`int`): The size for one batch, defaults to 1.
        """
        count = self._async_stream_send(iterator, batch_size)
        responses = self._get_data_responses(count)
        outputs = self._solve_responses(responses)
        return outputs

    def _async_stream_send(self, values, batch_size=1):
        count = 1
        for value in values:
            if not isinstance(value, list):
                value = [value]
            if len(value) > 1 and batch_size == 1:
                value = [value]

            inputs = self._solve_inputs(value, batch_size)
            self._client.async_stream_infer(model_name=self._model_name,
                                            inputs=inputs,
                                            request_id=str(count))
            count = count + 1
        return count

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):  # pylint: disable=redefined-builtin
        self.stop_stream()

    def stop_stream(self):
        self._client.stop_stream()

    def _get_data_responses(self, count):
        responses = []
        for _ in range(count - 1):
            res, error = self.user_data.completed_requests.get()
            if error is not None:
                raise error
            responses.append(res)
        return responses

    @staticmethod
    def _solve_inputs(args, size=1):
        from towhee.utils.tritonclient_utils import grpcclient  # pylint: disable=import-outside-toplevel
        inputs = [grpcclient.InferInput('INPUT0', [size, 1], 'BYTES')]
        arg_list = []
        for arg in args:
            arg_list.append([json.dumps(arg, cls=NumpyArrayEncoder)])
        data = np.array(arg_list, dtype=np.object_)
        inputs[0].set_data_from_numpy(data)
        return inputs

    @staticmethod
    def _solve_responses(responses):
        outputs = []
        for response in responses:
            res = response.as_numpy('OUTPUT0')[0]
            outputs.append(json.loads(res, cls=NumpyArrayDecoder))
        return outputs
