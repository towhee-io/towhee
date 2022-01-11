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
import shutil
from pathlib import Path

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
        self._temp = {'pipeline': 'pipeline-template'}

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
        url = 'https://hub.towhee.io/api/v1/user/repos'
        try:
            r = requests.post(url, data=data, headers={'Authorization': 'token ' + token_hash})
            r.raise_for_status()
        except HTTPError as e:
            self.delete_token(token_id, password)
            raise e

        self.delete_token(token_id, password)

    def init(self) -> None:
        """
        Initialize the file structure with template. First clone the repo, then initialize it with the template file structure.

        Raises:
            (`HTTPError`)
                Raise error in request.
            (`OSError`)
                Raise error in writing file.
        """
        self.clone(local_dir=self._repo, tag='main')
        # Download the template repo file
        repo_temp = self._temp['pipeline']
        git.Repo.clone_from(url=f'{self.root}/towhee/{repo_temp}.git', to_path=f'{self._repo}/{repo_temp}', branch='main')

        ori_str_list = [f'author/{repo_temp}', repo_temp, ''.join(x.title() for x in repo_temp.split('-'))]
        tar_str_list = [f'{self._author}/{self._repo}', self._repo, ''.join(x.title() for x in self._repo.split('-'))]
        for file in (Path(self._repo) / repo_temp).glob('*'):
            if file.name.endswith(('.md', '.yaml')):
                new_file = Path(self._repo) / str(file.name).replace(repo_temp.replace('-', '_'), self._repo.replace('-', '_'))
                self.update_text(ori_str_list, tar_str_list, file, new_file)
            elif file.name != '.git':
                os.rename(file, Path(self._repo) / file.name)
        shutil.rmtree(Path(self._repo) / repo_temp)
