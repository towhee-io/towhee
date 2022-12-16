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
# pylint: disable=import-outside-toplevel
# pylint: disable=unspecified-encoding

class DockerImageBuilder:
    '''
    Build triton image
    '''
    def __init__(self, towhee_pipeline: 'towhee.RuntimePipeline', image_name: str, server_config: dict, cuda_version: str):
        self._towhee_pipeline = towhee_pipeline
        self._image_name = image_name
        self._server_config = server_config
        self._cuda_version = cuda_version

    def prepare_dag(self, workspace: Path):
        from towhee.utils.thirdparty.dail_util import dill as pickle
        dag = self._towhee_pipeline.dag_repr
        with open(workspace / 'dag.pickle', 'wb') as f:
            pickle.dump(dag, f)

    def prepare_config(self, workspace: Path):
        config = self._server_config
        with open(workspace / 'server_config.json', 'w') as f:
            json.dump(config, f)

    def build_image(self, workspace: Path):
        cmd = 'cd {} && docker build -t {} .'.format(workspace, self._image_name)
        subprocess.run(cmd, shell=True, check=True)

    def docker_file(self) -> Path:
        return get_dockerfile(self._cuda_version)

    def build(self) -> bool:
        with TemporaryDirectory(dir='./') as workspace:
            self.prepare_dag(Path(workspace))
            self.prepare_config(Path(workspace))
            file_path = self.docker_file()
            if file_path is None:
                return False
            shutil.copy(file_path, Path(workspace) / 'Dockerfile')
            self.build_image(workspace)
            return True
