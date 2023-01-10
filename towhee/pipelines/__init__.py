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


from pathlib import Path

from towhee.utils.log import engine_log


BUILT_IN_PIPES_ROOT = Path(__file__).absolute().parent


def get_builtin_pipe_file(name: str) -> str:
    file_name = name + '.py'
    file_path = BUILT_IN_PIPES_ROOT / file_name
    if not file_path.is_file():
        engine_log.info('%s not a built-in pipeline', name)
        return None
    return file_path
