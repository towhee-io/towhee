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
    # pylint: disable=unused-import,ungrouped-imports
    import tritonclient.http as httpclient
    import tritonclient.http.aio as aio_httpclient
    from tritonclient.utils import InferenceServerException, np_to_triton_dtype
except ModuleNotFoundError as moduleNotFound:
    try:
        from towhee.utils.dependency_control import prompt_install
        prompt_install('tritonclient[http]==2.32.0')
        # pylint: disable=unused-import,ungrouped-imports
        import tritonclient.http as httpclient
        import tritonclient.http.aio as aio_httpclient
        from tritonclient.utils import InferenceServerException, np_to_triton_dtype
    except:
        from towhee.utils.log import engine_log
        engine_log.error('tritonclient not found, you can install via `pip install tritonclient[http]`.')
        raise ModuleNotFoundError('tritonclient not found, you can install via `pip install tritonclient[http]`.') from moduleNotFound
