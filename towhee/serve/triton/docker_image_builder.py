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
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from towhee.serve.triton.dockerfiles import get_dockerfile


class DockerImageBuilder:
    '''
    Build triton image
    '''
    def __init__(self, towhee_pipeline: 'towhee.dc', image_name: str, cuda: str):
        self._towhee_pipeline = towhee_pipeline
        self._image_name = image_name
        self._cuda = cuda

    def prepare_dag(self, workspace: Path):
        dag = self._towhee_pipeline.dag_info
        for v in dag.values():
            del v['call_args']
        with open(workspace / 'dag.json', 'wt', encoding='utf-8') as f:
            json.dump(dag, f)

    def build_image(self, workspace: Path):
        cmd = 'cd {} && docker build -t {} .'.format(workspace,
                                                     self._image_name)
        subprocess.run(cmd, shell=True, check=True)

    def docker_file(self) -> Path:
        return get_dockerfile(self._cuda)

    def build(self) -> bool:
        with TemporaryDirectory(dir='./') as workspace:
            self.prepare_dag(Path(workspace))
            file_path = self.docker_file()
            if file_path is None:
                return False
            shutil.copy(file_path, Path(workspace) / 'Dockerfile')
            self.build_image(workspace)
            return True
