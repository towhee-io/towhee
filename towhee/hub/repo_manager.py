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
import os
import subprocess
import sys
from typing import Tuple
from pathlib import Path

import git
from towhee.utils.log import engine_log
from towhee.hub.download_tools import obtain_lfs_extensions, latest_commit, get_file_list, download_files
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError


class RepoManager:
    """
    Base Repo Manager class to create, initialize, download repos.

    Args:
        author (`str`):
            The author of the repo.
        repo (`str`):
            The name of the repo.
    """
    def __init__(self, author: str, repo: str):
        self._author = author
        self._repo = repo
        self._type_dict = {'model': 1, 'operator': 2, 'pipeline': 3, 'dataset': 4}

    @property
    def author(self):
        return self._author

    @property
    def repo(self):
        return self._repo

    @property
    def type_dict(self):
        return self._type_dict

    def create_token(self, token_name: str, password: str) -> Tuple[int, str]:
        """
        Create an account verification token.

        This token allows for avoiding HttpBasicAuth for subsequent calls.

        Args:
            token_name (`str`):
                The name to be given to the token.
            password (`str`):
                The password of current author.

        Returns:
            (`Tuple[int, str]`)
                Return the token id and the sha-1.

        Raises:
            (`HTTPError`)
                Raise the error in request.
        """
        url = f'https://hub.towhee.io/api/v1/users/{self._author}/tokens'
        data = {'name': token_name}
        try:
            r = requests.post(url, data=data, auth=HTTPBasicAuth(self._author, password))
            r.raise_for_status()
        except HTTPError as e:
            raise e

        res = r.json()
        token_id = str(res['id'])
        token_sha1 = str(res['sha1'])

        return token_id, token_sha1

    def delete_token(self, token_id: int, password: str) -> None:
        """
        Delete the token with the given name. Useful for cleanup after changes.

        Args:
            token_id (`int`):
                The token id.
            password (`str`):
                The password of current author.
        """
        url = f'https://hub.towhee.io/api/v1/users/{self._author}/tokens/{token_id}'
        try:
            r = requests.delete(url, auth=HTTPBasicAuth(self._author, password))
            r.raise_for_status()
        except HTTPError as e:
            raise e

    def exists(self) -> bool:
        """
        Check if a repo exists.

        Returns:
            (`bool`)
                return `True` if the repository exists, else `False`.

        Raises:
            (`HTTPError`)
                Raise the error in request.
        """
        try:
            url = f'https://hub.towhee.io/api/v1/repos/{self._author}/{self._repo}'
            r = requests.get(url)
            return r.status_code == 200
        except HTTPError as e:
            raise e

    def clone(self, tag: str, local_dir: str, install_reqs: bool = True) -> None:
        """
        Performs a download of the selected repo to specified location.

        First checks to see if lfs is tracking files, then finds all the filepaths
        in the repo and lastly downloads them to the location.

        Args:
            tag (`str`):
                The tag name.
            local_dir (`str`):
                The local directory being downloaded to
            install_reqs (`bool`):
                Whether to install packages from requirements.txt

        Raises:
            (`HTTPError`)
                Raise error in request.
            (`OSError`)
                Raise error in writing file.
        """
        if not self.exists():
            engine_log.error('%s/%s repo does not exist.', self._author, self._repo)
            raise ValueError(f'{self._author}/{self._author} repo does not exist.')

        url = f'https://towhee.io/{self._author}/{self._repo}.git'
        git.Repo.clone_from(url=url, to_path=local_dir, branch=tag)

        if install_reqs:
            if 'requirements.txt' in os.listdir(local_dir):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', Path(local_dir) / 'requirements.txt'])

    def download(self, tag: str, local_dir: str):
        """
        Download repo without git.

        Agrs:
            tag (`str`):
                The tag of the repo to download.
        """
        if not self.exists():
            engine_log.error('%s/%s repo does not exist.', self._author, self._repo)
            raise ValueError(f'{self._author}/{self._author} repo does not exist.')

        lfs_files = obtain_lfs_extensions(self._author, self._repo, tag)
        commit = latest_commit(self._author, self._repo, tag)
        file_list = get_file_list(self._author, self._repo, commit)
        download_files(self._author, self._repo, tag, file_list, lfs_files, local_dir, False)

    def covert_dic(self, dicts: dict) -> dict:
        """
        Convert all the values in a dictionary to str and replace char.

        For example:
        <class 'torch.Tensor'>(unknow type) to torch.Tensor(str type).

        Args:
            dicts (`dict`):
                The dictionary to convert.

        Returns:
            (`dict`)
                The converted dictionary.
        """
        for keys in dicts:
            dicts[keys] = str(dicts[keys]).replace('<class ', '').replace('>', '').replace('\'', '')
        return dicts
