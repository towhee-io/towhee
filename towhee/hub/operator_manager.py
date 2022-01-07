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
import subprocess
import sys
import yaml
from pathlib import Path
from importlib import import_module

from towhee.hub.download_tools import obtain_lfs_extensions, latest_commit, get_file_list, download_files
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
    def __init__(self, author: str, repo: str):
        super().__init__(author, repo)
        self._repo_type = 'operator'

    def create(self, password: str, private: bool = True) -> None:
        """
        Create a repo under the account connected to the given token.

        Args:
            password (`str`):
                Current author's password.
            private (`bool`):
                If the repo is private.

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
            'default_tag': 'main',
            'description': 'This is another operator repo',
            'name': self._repo,
            'private': private,
            'template': False,
            'trust_model': 'default',
            'type': self._type_dict[self._repo_type]
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
        Initialize the repo with template.

        First clone the repo, then download and rename the template repo file.

        Raises:
            (`HTTPError`)
                Raise error in request.
            (`OSError`)
                Raise error in writing file.
        """
        links = 'https://hub.towhee.io/' + self._author + '/' + self._repo + '.git'
        subprocess.call(['git', 'clone', links])

        repo_file_name = self._repo.replace('-', '_')

        # TODO: distinguish nnop and pyop (Shiyu)
        lfs_files = obtain_lfs_extensions('towhee', 'operator-template', 'main')
        commit = latest_commit('towhee', 'operator-template', 'main')
        file_list = get_file_list('towhee', 'operator-template', commit)
        download_files('towhee', 'operator-template', 'main', file_list, lfs_files, str(Path.cwd() / self._repo), False)

        (Path(self._repo) / 'operator_template.py').rename(Path(self._repo) / (repo_file_name + '.py'))
        (Path(self._repo) / 'operator_template.yaml').rename(Path(self._repo) / (repo_file_name + '.yaml'))

    def generate_yaml(self) -> None:
        """
        Generate the yaml of Operator.

        Example:
            name: 'operator-template'
            labels:
            recommended_framework: ''
            class: ''
            others: ''
            operator: 'towhee/operator-template'
            init:
            model_name: str
            call:
            input:
                img_path: str
            output:
                feature_vector: numpy.ndarray

        Raises:
            (`HTTPError`)
                Raise error in request.
            (`OSError`)
                Raise error in writing file.
        """
        sys.path.append(self._repo)
        repo_file_name = self._repo.replace('-', '_')
        # get class name in camel case
        components = self._repo.split('-')
        class_name = ''.join(x.title() for x in components)
        yaml_file = self._repo + '/' + repo_file_name + '.yaml'
        operator_name = self._author + '/' + self._repo
        # import the class from repo
        cls = getattr(import_module(repo_file_name, self._repo), class_name)

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
        call_input = call_func

        data = {
            'name': self._repo,
            'labels': {
                'recommended_framework': '', 'class': '', 'others': ''
            },
            'operator': operator_name,
            'init': self.covert_dic(init_args),
            'call': {
                'input': self.covert_dic(call_input), 'output': self.covert_dic(call_output)
            }
        }
        with open(yaml_file, 'w', encoding='utf-8') as outfile:
            yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)
