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
import sys
import os
import yaml
from pathlib import Path
from typing import Union
from shutil import copytree, copyfile
from importlib import import_module

import git
from towhee.hub.repo_manager import RepoManager
from towhee.utils.log import engine_log
from requests.exceptions import HTTPError


class OperatorManager(RepoManager):
    """
    The Repo Manager to manage the operator repos.

    Args:
        author (`str`):
            The author of the repo.
        repo (`str`):
            The name of the repo.
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
            return None

        token_name = random.randint(0, 10000)
        token_id, token_hash = self.create_token(token_name, password)

        data = {
            'auto_init': True,
            'default_branch': 'main',
            'description': 'This is another operator repo',
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

    def init(self, is_nn: bool, file_src: Union[str, Path], file_dst: Union[str, Path] = None) -> None:
        """
        Initialize the repo with template.

        First clone the repo, then move and rename the template repo file.

        Args:
            is_nn (`bool`):
                If the operator is an nnoperator(neural network related).
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
            file_dst = Path().cwd() / repo_file_name
        file_src = Path(file_src)
        file_dst = Path(file_dst)

        url = self._root + '/' + self._author + '/' + self._repo + '.git'
        git.Repo.clone_from(url=url, to_path=file_dst, branch='main')

        if is_nn:
            template = 'nnoperator_template'
        else:
            template = 'pyoperator_template'

        for f in os.listdir(file_src):
            if (file_dst / f).is_file() or (file_dst / f).is_dir():
                continue
            if (file_src / f).is_file():
                copyfile(file_src / f, file_dst / f)
            elif (file_src / f).is_dir():
                copytree(file_src / f, file_dst / f)

        (file_dst / (template + '.py')).rename(file_dst / (repo_file_name + '.py'))
        (file_dst / (template + '.yaml')).rename(file_dst / (repo_file_name + '.yaml'))

    def generate_yaml(self) -> None:
        """
        Generate the yaml of Operator.

        Raises:
            (`HTTPError`)
                Raise error in request.
            (`OSError`)
                Raise error in writing file.
        """
        sys.path.append(str(Path.cwd()))
        yaml_file = Path(self._repo.replace('-', '_') + '.yaml')
        if yaml_file.exists():
            print(f'There already have {yaml_file}, please remove it first.')
            sys.exit()

        class_name = ''.join(x.title() for x in self._repo.split('-'))
        author_operator = self._author + '/' + self._repo
        # import the class from repo
        cls = getattr(import_module('.', self._repo.replace('-', '_')), class_name)
        init_args = cls.__init__.__annotations__
        try:
            del init_args['return']
        except KeyError:
            pass
        call_func = cls.__call__.__annotations__
        try:
            call_output = call_func.pop('return')
            call_output = call_output.__annotations__
        except KeyError:
            pass

        data = {
            'name': self._repo,
            'labels': {
                'recommended_framework': '', 'class': '', 'others': ''
            },
            'operator': author_operator,
            'init': self.covert_dic(init_args),
            'call': {
                'input': self.covert_dic(call_func), 'output': self.covert_dic(call_output)
            }
        }
        with open(yaml_file, 'a', encoding='utf-8') as outfile:
            yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)
