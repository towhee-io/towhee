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

class RemoteMixin:
    '''
    Mixin for triton

    Parameters:
        url: str
            triton server ip and port.the most likely url will be "127.0.0.1:8001".
        mode: str
            The function of inference.
            'grpc' protocol has 'infer'/'async_set'/'stream' functions.
            'http' protocol only has 'infer' function.
            The default value is 'infer'.
        model_name: str
            The name of the model to run inference.
            The default value is 'pipline'.
        protocol: str
            Communication protocol between triton server and client.
            The default value is 'grpc'.

    '''
    def remote(self, url, mode='infer', model_name='pipeline', protocol='grpc'):
        from towhee.serve.triton.client import Client
        from towhee.utils.tritonclient_utils import InferenceServerException
        self.triton_client = Client.init(url, model_name=model_name)
        try:
            if protocol == 'grpc' and (mode in ('infer', 'async_infer')):
                self.triton_client = Client.init(url, model_name=model_name, stream=False)
                if mode == 'infer':
                    res, err = self.triton_client.infer(list(self))
                else:
                    res, err = self.triton_client.async_infer(list(self))
            elif protocol == 'grpc' and mode == 'stream':
                self.triton_client = Client.init(url, model_name=model_name, stream=True, protocol=protocol)
                res, err = self.triton_client.stream(iter(self))
                self.triton_client.stop_stream()
            elif protocol == 'http' and mode == 'infer':
                self.triton_client = Client.init(url, model_name=model_name, stream=False, protocol=protocol)
                res, err = self.triton_client.infer(list(self))
            else:
                raise TypeError("Http protocol doesn't have this mode")
        except (TypeError, InferenceServerException) as e:
            return None, e

        return res, err
