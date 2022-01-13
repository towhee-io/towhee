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

import requests
import random
import os
from pathlib import Path
from typing import Union
from shutil import copytree, copyfile

import git
from towhee.hub.repo_manager import RepoManager
from towhee.utils.log import engine_log
from requests.exceptions import HTTPError


class PipelineManager(RepoManager):
    """
    The Repo Manager to manage the pipeline repos.

    Args:
        author (`str`):
            The author of the repo.
        repo (`str`):
            The name of the repo.
    """
    def __init__(self, author: str, repo: str, root: str = 'https://hub.towhee.io'):
        super().__init__(author, repo, root)
        # 3 represents pipelines when creating a repo in Towhee's hub
        self._class = 3

    def create(self, password: str) -> None:
        """
        Create a repo under current account.

        Args:
            password (`str`):
                Current author's password.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        if self.exists():
            engine_log.info('%s/%s repo already exists.', self._author, self._repo)
            return None

        token_name = random.randint(0, 10000)
        token_id, token_hash = self.create_token(token_name, password)

        data = {
            'auto_init': True,
            'default_branch': 'main',
            'description': 'This is another pipeline repo',
            'name': self._repo,
            'private': False,
            'template': False,
            'trust_model': 'default',
            'type': self._class
        }
        url = self._root + '/api/v1/user/repos'
        try:
            r = requests.post(url, data=data, headers={'Authorization': 'token ' + token_hash})
            r.raise_for_status()
        except HTTPError as e:
            self.delete_token(token_id, password)
            raise e

        self.delete_token(token_id, password)

    def init(self, file_src: Union[str, Path], file_dst: Union[str, Path] = None) -> None:
        """
        Initialize the repo with template.

        First clone the repo, then move and rename the template repo file.

        Args:
            file_src (`Union[str, Path]`):
                The path to the template files.
            file_dst (`Union[str, Path]`):
                The path to the local repo to init.

        Raises:
            (`HTTPError`)
                Raise error in request.
            (`OSError`)
                Raise error in writing file.
        """
        repo_file_name = self._repo.replace('-', '_')

        if not file_dst:
            file_dst = Path.cwd() / repo_file_name
        file_src = Path(file_src)
        file_dst = Path(file_dst)

        url = self._root + '/' + self._author + '/' + self._repo + '.git'
        git.Repo.clone_from(url=url, to_path=file_dst, branch='main')

        for f in os.listdir(file_src):
            if (file_dst / f).is_file() or (file_dst / f).is_dir():
                continue
            if (file_src / f).is_file():
                copyfile(file_src / f, file_dst / f)
            elif (file_src / f).is_dir():
                copytree(file_src / f, file_dst / f)

        (file_dst / 'pipeline_template.yaml').rename(file_dst / (repo_file_name + '.yaml'))
