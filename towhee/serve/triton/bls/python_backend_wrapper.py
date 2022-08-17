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


try:
    # In triton env
    import triton_python_backend_utils as pb_utils  # pylint: disable=unused-import
except Exception:  # pylint: disable=broad-except
    # In test env
    from towhee.serve.triton.bls.mock.mock_pb_util import MockTritonPythonBackendUtils
    pb_utils = MockTritonPythonBackendUtils  # pylint: disable=invalid-name
