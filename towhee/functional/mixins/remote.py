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

from towhee.serve.triton.client import *

class RemoteMixin:
    '''
    Mixin for triton
    '''
    def remote(self, url, mode='infer', model_name='pipeline', protocol='grpc'):
        tritonClient = Client.init(url, model_name=model_name)
        try:
            if protocol == 'grpc' and (mode == 'infer' or mode == 'async_infer'):
                tritonClient = Client.init(url, model_name=model_name, stream=False)
                if mode == 'infer':
                    res, err = tritonClient.infer(list(self))
                else:
                    res, err = tritonClient.async_infer(list(self))
            elif protocol == 'grpc' and mode == 'stream':
                tritonClient = Client.init(url, model_name=model_name, stream=True, protocol=protocol)
                res, err = tritonClient.stream(iter(self))
                tritonClient.stop_stream()
            elif protocol == 'http' and mode == 'infer':
                tritonClient = Client.init(url, model_name=model_name, stream=False, protocol=protocol)
                res, err = tritonClient.infer(list(self))
            else:
                raise TypeError("Http protocol doesn't have this mode")
        except TypeError as e:
            return None, e

        return res, err