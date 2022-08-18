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

import os
import sys
from typing import Union
from pathlib import Path
from importlib import import_module

from towhee.hub.repo_manager import RepoManager
from towhee.utils.log import engine_log
from towhee.utils.yaml_utils import load_yaml, dump_yaml


class OperatorManager(RepoManager):
    """
    The Repo Manager to manage the operator repos.

    Args:
        author (`str`):
            The author of the repo.
        repo (`str`):
            The name of the repo.
        root (`str`):
            The root url where the repo located.
    """
    def __init__(self, author: str, repo: str, root: str = 'https://hub.towhee.io'):
        super().__init__(author, repo, root)
        # 2 represents operators when creating a repo in Towhee's hub
        self._class = 2

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
        else:
            self.hub_utils.create(password, self._class)

    def create_with_token(self, token: str) -> None:
        """
        Create a repo under current account.

        Args:
            token (`str`):
                Current author's token.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        if self.exists():
            engine_log.info('%s/%s repo already exists.', self._author, self._repo)
        else:
            self.hub_utils.create_repo(token, self._class)

    def init_nnoperator(self, file_temp: Union[str, Path], file_dest: Union[str, Path], framework: str = 'pytorch') -> None:
        """
        Initialize the files under file_dest by moving and updating the text under file_temp.

        Args:
            file_temp (`Union[str, Path]`):
                The path to the template files.
            file_dest (`Union[str, Path]`):
                The path to the local repo to init.
            framework (`str, Path`):
                The framework for the model, defaults to 'pytorch'

        Raises:
            (`HTTPError`)
                Raise error in request.
            (`OSError`)
                Raise error in writing file.
        """
        repo_temp = self._temp['nnoperator']

        ori_str_list = [f'author/{repo_temp}', repo_temp, ''.join(x.title() for x in repo_temp.split('-')), 'pytorch']
        tar_str_list = [f'{self._author}/{self._repo}', self._repo, ''.join(x.title() for x in self._repo.split('-')), framework]
        for file in Path(file_temp).glob('*'):
            if file.name.endswith(('.md', '.yaml', 'template.py', '__init__.py')):
                new_file = Path(file_dest) / str(file.name).replace(repo_temp.replace('-', '_'), self._repo.replace('-', '_'))
                self.hub_utils.update_text(ori_str_list, tar_str_list, str(file), str(new_file))
            elif file.name != '.git':
                os.rename(file, Path(file_dest) / file.name)
        if framework != 'pytorch':
            os.rename(Path(file_dest) / 'pytorch', Path(file_dest) / framework)

    def init_pyoperator(self, file_temp: Union[str, Path], file_dest: Union[str, Path]) -> None:
        """
        Initialize the files under file_dest by moving and updating the text under file_temp.

        Args:
            file_temp (`Union[str, Path]`):
                The path to the template files.
            file_dest (`Union[str, Path]`):
                The path to the local repo to init.

        Raises:
            (`HTTPError`)
                Raise error in request.
            (`OSError`)
                Raise error in writing file.
        """
        repo_temp = self._temp['pyoperator']

        temp_module = repo_temp.replace('-', '_')
        repo_module = self._repo.replace('-', '_')

        ori_str_list = [f'namespace.{temp_module}', temp_module, repo_temp, ''.join(x.title() for x in repo_temp.split('-'))]
        tar_str_list = [f'{self._author}.{repo_module}', repo_module, self._repo, ''.join(x.title() for x in self._repo.split('-'))]
        for file in Path(file_temp).glob('*'):
            if file.name.endswith(('.md', '.yaml', 'template.py', '__init__.py')):
                new_file = Path(file_dest) / str(file.name).replace(temp_module, repo_module)
                self.hub_utils.update_text(ori_str_list, tar_str_list, str(file), str(new_file))
            elif file.name != '.git':
                os.rename(file, Path(file_dest) / file.name)
