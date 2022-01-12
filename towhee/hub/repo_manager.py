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
import re
import subprocess
import sys
from typing import Tuple, List, Union
from pathlib import Path

import git
from tempfile import TemporaryFile
from towhee.hub.download_tools import Worker
from towhee.utils.log import engine_log
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
    def __init__(self, author: str, repo: str, root: str = 'https://hub.towhee.io'):
        self._author = author
        self._repo = repo
        self._root = root

    @property
    def author(self):
        return self._author

    @property
    def repo(self):
        return self._repo

    @property
    def root(self):
        return self._root

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
        url = f'{self.root}/api/v1/users/{self._author}/tokens'
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
        url = f'{self.root}/api/v1/users/{self._author}/tokens/{token_id}'
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
            url = f'{self.root}/api/v1/repos/{self._author}/{self._repo}'
            r = requests.get(url)
            return r.status_code == 200
        except HTTPError as e:
            raise e

    def clone(self, local_dir: Union[str, Path], tag: str = 'main', install_reqs: bool = True) -> None:
        """
        Performs a download of the selected repo to specified location.

        First checks to see if lfs is tracking files, then finds all the filepaths
        in the repo and lastly downloads them to the location.

        Args:
            tag (`str`):
                The tag name.
            local_dir (`Union[str, Path]`):
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

        url = f'{self.root}/{self._author}/{self._repo}.git'
        git.Repo.clone_from(url=url, to_path=local_dir, branch=tag)

        if install_reqs:
            if 'requirements.txt' in os.listdir(local_dir):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', Path(local_dir) / 'requirements.txt'])

    def obtain_lfs_extensions(self, author: str, repo: str, tag: str) -> List[str]:
        """
        Download the .gitattributes file from the specified repo in order to figure out
        which files are being tracked by git-lfs.

        Lines that deal with git-lfs take on the following format:

        ```
            *.extension   filter=lfs  merge=lfs ...
        ```

        Args:
            author (`str`):
                The account name.
            repo (`str`):
                The repo name.
            tag (`str`):
                The tag name.

        Returns:
            (`List[str]`)
                The list of file extentions tracked by git-lfs.
        """
        url = f'{self.root}/api/v1/repos/{author}/{repo}/raw/.gitattributes?ref={tag}'
        lfs_files = []

        # Using temporary file in order to avoid double download, cleaner to not split up downloads everywhere.
        with TemporaryFile() as temp_file:
            try:
                r = requests.get(url)
                r.raise_for_status()
            except HTTPError:
                return lfs_files

            temp_file.write(r.content)
            temp_file.seek(0)

            for line in temp_file:
                parts = line.split()
                # We only care if lfs filter is present.
                if b'filter=lfs' in parts[1:]:
                    # Removing the `*` in `*.ext`, need work if filtering specific files.
                    lfs_files.append(parts[0].decode('utf-8')[1:])

        return lfs_files

    def latest_commit(self, author: str, repo: str, tag: str) -> str:
        """
        Grab the latest commit of a tag.

        Args:
            author (`str`):
                The account name.
            repo (`str`):
                The repo name.
            tag (`str`):
                The tag name.

        Returns:
            (`str`)
                The latest commit hash cut down to 10 characters.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """

        url = f'{self.root}/api/v1/repos/{author}/{repo}/commits?limit=1&page=1&sha={tag}'
        try:
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status()
        except HTTPError as e:
            raise e

        res = r.json()

        return res[0]['sha'][:10]

    def get_file_list(self, author: str, repo: str, commit: str) -> List[str]:
        """
        Get all the files in the current repo at the given commit.

        This is done through forming a git tree recursively and filtering out all the files.

        Args:
            author (`str`):
                The account name.
            repo (`str`):
                The repo name.
            commit (`str`):
                The commit to base current existing files.

        Returns:
            (`List[str]`)
                The file paths for the repo

        Raises:
            (`HTTPError`)
                Raise error in request.
        """

        url = f'{self.root}/api/v1/repos/{author}/{repo}/git/trees/{commit}?recursive=1'
        file_list = []
        try:
            r = requests.get(url)
            r.raise_for_status()
        except HTTPError as e:
            raise e

        res = r.json()
        # Check each object in the tree
        for file in res['tree']:
            # Ignore directories (they have the type 'tree')
            if file['type'] != 'tree':
                file_list.append(file['path'])

        return file_list

    def download_files(
        self,
        author: str,
        repo: str,
        tag: str,
        file_list: List[str],
        lfs_files: List[str],
        local_dir: Union[str, Path],
        install_reqs: bool = True
    ) -> None:
        """
        Download the files from hub. One url is used for git-lfs files and another for the other files.

        Args:
            author (`str`):
                The account name.
            repo (`str`):
                The repo name.
            tag (`str`):
                The tag name.
            file_list (`List[str]`):
                The hub file paths.
            lfs_files (`List[str]`):
                The file extensions being tracked by git-lfs.
            local_dir (`Union[str, Path]`):
                The local directory to download to.
            install_reqs (`bool`):
                Whether to install packages from requirements.txt

        Raises:
            (`HTTPError`)
                Rasie error in request.
            (`OSError`)
                Raise error in writing file.
        """
        threads = []

        # If the trailing forward slash is missing, add it on.
        if isinstance(local_dir, Path):
            local_dir = str(local_dir)

        if local_dir[-1] != '/':
            local_dir += '/'

        # endswith() can check multiple suffixes if they are a tuple.
        lfs_files = tuple(lfs_files)

        for file_name in file_list:
            # Files dealt with lfs have a different url.
            if file_name.endswith(lfs_files):
                url = f'{self.root}/{author}/{repo}/media/branch/{tag}/{file_name}'
            else:
                url = f'{self.root}/api/v1/repos/{author}/{repo}/raw/{file_name}?ref={tag}'

            threads.append(Worker(url, local_dir, file_name))
            threads[-1].start()

        for thread in threads:
            thread.join()

        if install_reqs:
            requirements = list(filter(lambda x: re.match(r'(.*/)?requirements.txt', x) is not None, file_list))
            for req in requirements:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', local_dir + req])

    def download(self, local_dir: Union[str, Path], tag: str = 'main'):
        """
        Download repo without git.

        Agrs:
            local_dir (`Union[str, Path]`):
                Thre local dir to download the files into.
            tag (`str`):
                The tag of the repo to download.
        """
        if not self.exists():
            engine_log.error('%s/%s repo does not exist.', self._author, self._repo)
            raise ValueError(f'{self._author}/{self._author} repo does not exist.')

        lfs_files = self.obtain_lfs_extensions(self.author, self.repo, tag)
        commit = self.latest_commit(self.author, self.repo, tag)
        file_list = self.get_file_list(self.author, self.repo, commit)
        self.download_files(self.author, self.repo, tag, file_list, lfs_files, local_dir, False)

    @staticmethod
    def convert_dict(dicts: dict) -> dict:
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
        return dict(dicts)

    def update_text(self, ori_str_list: list, tar_str_list: list, file: str, new_file: str) -> None:
        """
            Update the text in the file and rename it with the new file name.
            Args:
                ori_str_list (`list`):
                    The original str list to be replaced
                tar_str_list (`list`):
                    The target str list after replace
                file (`str`):
                    The original file name to be updated
                new_file (`str`):
                    The target file name after update
        """
        with open(file, 'r', encoding='utf-8') as f1:
            file_text = f1.read()
        # Replace the target string
        for ori_str, tar_str in zip(ori_str_list, tar_str_list):
            file_text = file_text.replace(ori_str, tar_str)
        with open(new_file, 'w', encoding='utf-8') as f2:
            f2.write(file_text)
