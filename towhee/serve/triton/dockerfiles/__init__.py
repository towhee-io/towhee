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

import logging

logger = logging.getLogger()


def get_dockerfile(cuda_version: str) -> Path:
    if cuda_version == '11.3':
        file_name = 'DockerfileCuda113'
    elif cuda_version == '11.4':
        file_name = 'DockerfileCuda114'
    elif cuda_version == '11.6':
        file_name = 'DockerfileCuda116'
    elif cuda_version == '11.7':
        file_name = 'DockerfileCuda117'
    elif cuda_version == '117dev':
        # for QA to test towhee
        file_name = 'DockerfileCuda117dev'
    else:
        logger.error("Towhee serve doesn't support cuda %s, the support cuda: 11.3, 11.4, 11.6, 11.7", cuda_version)
        return None
    return Path(__file__).absolute().parent / file_name
