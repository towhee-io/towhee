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

import queue
import numpy as np
from towhee.utils.tritonclient_utils import grpcclient, httpclient, InferenceServerException
from functools import partial

class UserData:
    def __init__(self):
        self.completed_requests = queue.Queue()

# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data.completed_requests.put((result, error))


class Client():
    @staticmethod
    def init(url, model_name='pipeline', stream=False, protocol='grpc'):
        if protocol == 'http':
            return HttpClient(url, model_name)
        else:
            if stream is True:
                return GrpcStreamClient(url, model_name)
            else:
                return GrpcClient(url, model_name)


class HttpClient():
    '''
    triton HttpClient

    Args:
        url(`string`):
            triton server's ip
            example: '127.0.0.1'
        model_name(`string`):
            model name
            example: 'pipeline'
    '''
    def __init__(self, url, model_name):
        self.client = httpclient.InferenceServerClient(url)
        self._model_name = model_name

    def get_model_name(self):
        return self._model_name

    def infer(self, arg):
        if isinstance(arg, list) is False:
            arg = [arg]
        inputs, outputs = self._solve_inputs_outputs(arg)

        sent_count = 0
        responses = []
        for _ in range(len(arg)):
            sent_count += 1
            responses.append(self.client.infer(self.get_model_name(), inputs, request_id=str(sent_count), outputs=outputs))
        return self._solve_responses(outputs, responses)

    def _solve_inputs_outputs(self, arg):
        model_metadata = self.client.get_model_metadata(model_name=self.get_model_name())
        inputs = []
        inputs_list = model_metadata['inputs']
        outputs_list = model_metadata['outputs']
        for i in inputs_list:
            inputs.append(httpclient.InferInput(i['name'], i['shape'], i['datatype']))

        image_data = []
        for idx in range(len(arg)):
            image_data.append(np.array(arg[idx].encode('utf-8'), dtype=np.object_))
        if len(arg) > 1:
            batch_image_data = np.stack([image_data[0]], axis=0)
        else:
            batch_image_data = np.stack(image_data, axis=0)
        inputs[0].set_data_from_numpy(batch_image_data, binary_data=True)

        output_names = [
            output['name']
            for output in outputs_list
        ]

        outputs = []
        for output_name in output_names:
            outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=True))
        return inputs, outputs

    def _solve_responses(self, outputs, responses):
        res = []
        for i in range(len(responses)):
            res_dict = {}
            for j in range(len(outputs)):
                output_name = outputs[j].name()
                res_dict[output_name] = responses[i].as_numpy(output_name)
            res.append(res_dict)
        return res, None


class GrpcClient():
    '''
    triton GrpcClient

    Args:
        url(`string`):
            triton server's ip
            example: '127.0.0.1'
        model_name(`string`):
            model name
            example: 'pipeline'
    '''
    def __init__(self, url, model_name):
        self.client = grpcclient.InferenceServerClient(url)
        self.user_data = UserData()
        self._model_name = model_name

    def get_model_name(self):
        return self._model_name

    def async_infer(self, list_arg):
        inputs, outputs = self._solve_inputs_outputs(list_arg)
        sent_count = 0
        responses = []
        for _ in range(len(list_arg)):
            sent_count += 1
            self.client.async_infer(self.get_model_name(), inputs,\
            partial(completion_callback, self.user_data), request_id=str(sent_count), outputs=outputs)

        processed_count = 0
        while processed_count < sent_count:
            (results, error) = self.user_data.completed_requests.get()
            processed_count += 1
            err_dict = {}
            if error is not None:
                err_dict[processed_count] = error
            if processed_count <= len(list_arg):
                responses.append(results)

        return self._solve_responses(outputs, responses, error_dict=err_dict)

    def infer(self, arg):
        if isinstance(arg, list) is False:
            arg = [arg]
        inputs, outputs = self._solve_inputs_outputs(arg)

        sent_count = 0
        responses = []
        for _ in range(len(arg)):
            sent_count += 1
            responses.append(self.client.infer(self.get_model_name(), inputs, request_id=str(sent_count), outputs=outputs))
        return self._solve_responses(outputs, responses)

    def _solve_inputs_outputs(self, iterator):
        model_metadata = self.client.get_model_metadata(model_name=self.get_model_name())
        inputs = []
        inputs_list = model_metadata.inputs
        outputs_list = model_metadata.outputs
        for i in inputs_list:
            inputs.append(grpcclient.InferInput(i.name, i.shape, i.datatype))

        image_data = []
        for idx in range(len(iterator)):
            image_data.append(np.array(iterator[idx].encode('utf-8'), dtype=np.object_))
        if len(iterator) > 1:
            batch_image_data = np.stack([image_data[0]], axis=0)
        else:
            batch_image_data = np.stack(image_data, axis=0)
        inputs[0].set_data_from_numpy(batch_image_data)

        output_names = []
        for output in outputs_list:
            output_names.append(output.name)

        outputs = []
        for output_name in output_names:
            outputs.append(grpcclient.InferRequestedOutput(output_name))
        return inputs, outputs

    def _solve_responses(self, outputs, responses, error_dict=None):
        res = []
        for i in range(len(responses)):
            res_dict = {}
            for j in range(len(outputs)):
                output_name = outputs[j].name()
                res_dict[output_name] = responses[i].as_numpy(output_name)
            res.append(res_dict)
        return res, error_dict


class GrpcStreamClient():
    '''
    triton GrpcStreamClient

    Args:
        url(`string`):
            triton server's ip
            example: '127.0.0.1'
        model_name(`string`):
            model name
            example: 'pipeline'
    '''
    def __init__(self, url, model_name):
        self.client = grpcclient.InferenceServerClient(url)
        self.user_data = UserData()
        self.client.start_stream(partial(completion_callback, self.user_data))
        self._model_name = model_name

    def __del__(self):
        self.stop_stream()

    def get_model_name(self):
        return self._model_name

    def stream(self, iterator):
        sent_count = 0
        responses = []
        err_dict = {}
        while True:
            sent_count += 1
            try:
                info = next(iterator)
                if isinstance(info, list) is False:
                    info = [info]
                inputs, outputs = self._solve_inputs_outputs(info)
                self.client.async_stream_infer(self.get_model_name(), inputs, request_id=str(sent_count), outputs= outputs)
            except InferenceServerException as e:
                err_dict[sent_count] = e
            except StopIteration:
                break

        processed_count = 0
        server_error_count = len(err_dict)
        while processed_count < sent_count - server_error_count - 1:
            (results, error) = self.user_data.completed_requests.get()
            processed_count += 1
            if error is not None:
                err_dict[processed_count+server_error_count] = error
            responses.append(results)
        return self._solve_responses(outputs, responses, error_dict=err_dict)

    def _solve_inputs_outputs(self, item):
        model_metadata = self.client.get_model_metadata(model_name=self.get_model_name())
        inputs = []
        inputs_list = model_metadata.inputs
        outputs_list = model_metadata.outputs
        for i in inputs_list:
            inputs.append(grpcclient.InferInput(i.name, i.shape, i.datatype))

        image_data = []
        for idx in range(len(item)):
            image_data.append(np.array(item[idx].encode('utf-8'), dtype=np.object_))
        if len(item) > 1:
            batch_image_data = np.stack([image_data[0]], axis=0)
        else:
            batch_image_data = np.stack(image_data, axis=0)
        inputs[0].set_data_from_numpy(batch_image_data)

        output_names = []
        for output in outputs_list:
            output_names.append(output.name)

        outputs = []
        for output_name in output_names:
            outputs.append(grpcclient.InferRequestedOutput(output_name))
        return inputs, outputs

    def _solve_responses(self, outputs, responses, error_dict=None):
        res = []
        for i in range(len(responses)):
            res_dict = {}
            for j in range(len(outputs)):
                output_name = outputs[j].name()
                res_dict[output_name] = responses[i].as_numpy(output_name)
            res.append(res_dict)
        return res, error_dict

    def stop_stream(self):
        self.client.stop_stream()
