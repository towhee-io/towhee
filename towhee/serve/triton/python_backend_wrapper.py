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


class MockTritonPythonBackendUtils:
    '''
    mock triton_python_backend_utils, used in UT.
    '''
    @staticmethod
    def get_input_tensor_by_name(r, input_key):
        '''
        Args:
            r: dict
            input_key: str
        return:
            MockTritonPythonBackendTensor
        '''
        return r.get(input_key)


class MockTritonPythonBackendTensor:
    '''
    Mock python_backend tensor object.
    '''
    def __init__(self, data: 'ndarray'):
        self._data = data

    def as_numpy(self):
        return self._data


try:
    # in triton image
    import triton_python_backend_utils as pb_utils
except Exception:
    # in test env
    pb_utils = MockTritonPythonBackendUtils
