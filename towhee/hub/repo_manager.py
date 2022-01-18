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
import re
import subprocess
import sys
import time
from typing import Tuple, List, Union
from shutil import rmtree, move
from pathlib import Path
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError

from tqdm import tqdm
from tempfile import TemporaryFile
from concurrent.futures import ThreadPoolExecutor
from towhee.utils.log import engine_log


class RepoManager:
    """
    Base Repo Manager class to create, initialize, download repos.

    Args:
        author (`str`):
            The author of the repo.
        repo (`str`):
            The name of the repo.
        root (`str`):
            The root url of the repo.
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
        url = f'{self._root}/api/v1/users/{self._author}/tokens'
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
        url = f'{self._root}/api/v1/users/{self._author}/tokens/{token_id}'
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
            url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}'
            r = requests.get(url)
            return r.status_code == 200
        except HTTPError as e:
            raise e

    def obtain_lfs_extensions(self, tag: str) -> List[str]:
        """
        Download the .gitattributes file from the specified repo in order to figure out
        which files are being tracked by git-lfs.

        Lines that deal with git-lfs take on the following format:

        ```
            *.extension   filter=lfs  merge=lfs ...
        ```

        Args:
            tag (`str`):
                The tag name.

        Returns:
            (`List[str]`)
                The list of file extentions tracked by git-lfs.
        """
        url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}/raw/.gitattributes?ref={tag}'
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

    def latest_commit(self, tag: str) -> str:
        """
        Grab the latest commit of a tag.

        Args:
            tag (`str`):
                The tag name.

        Returns:
            (`str`)
                The latest commit hash cut down to 10 characters.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """

        url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}/commits?limit=1&page=1&sha={tag}'
        try:
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status()
        except HTTPError as e:
            raise e

        res = r.json()

        return res[0]['sha'][:10]

    def get_file_list(self, commit: str) -> List[str]:
        """
        Get all the files in the current repo at the given commit.

        This is done through forming a git tree recursively and filtering out all the files.

        Args:
            commit (`str`):
                The commit to base current existing files.

        Returns:
            (`List[str]`)
                The file paths for the repo

        Raises:
            (`HTTPError`)
                Raise error in request.
        """

        url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}/git/trees/{commit}?recursive=1'
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

    def download_executor(self, tag: str, file_name: str, lfs_files: Tuple[str], local_dir: Union[str, Path]) -> bool:
        """
        Load the content from url and write into local files.

        Args:
            tag (`str`):
                The tag name.
            file_list (`List[str]`):
                The hub file paths.
            lfs_files (`List[str]`):
                The file extensions being tracked by git-lfs.
            local_dir (`Union[str, Path]`):
                The local directory to download to.

        Raises:
            (`HTTPError`)
                Rasie error in request.
            (`OSError`)
                Raise error in writing file.
        """
        if file_name.endswith(lfs_files):
            url = f'{self._root}/{self._author}/{self._repo}/media/branch/{tag}/{file_name}'
        else:
            url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}/raw/{file_name}?ref={tag}'

        file_path = Path(local_dir) / file_name

        if not file_path.parent.resolve().exists():
            try:
                file_path.parent.resolve().mkdir()
            except FileExistsError:
                pass
            except OSError as e:
                raise e

        # Get content.
        for i in range(5):
            try:
                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    break
                elif r.status_code == 429 and i < 4:
                    time.sleep(2)
                    continue
                r.raise_for_status()
            except HTTPError as e:
                raise e

        # Create local files.
        file_size = int(r.headers.get('content-length', 0))
        chunk_size = 1024
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f'Downloading {file_name}')
        with open(file_path, 'wb') as local_file:
            for chunk in r.iter_content(chunk_size=chunk_size):
                local_file.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()

    def download_files(self, tag: str, file_list: List[str], lfs_files: List[str], local_dir: Union[str, Path], install_reqs: bool):
        """
        Download the given files.

        Agrs:
            tag (`str`):
                The tag of the repo to download.
            file_list (List[str]):
                The files to download
            lfs_files (List[str]):
                The lfs files extensions.
            local_dir (`Union[str, Path]`):
                Thre local dir to download the files into.
            install_reqs (`bool`):
                Whether to install packages from requirements.txt.
        """
        if not local_dir:
            local_dir = Path.cwd()
        local_dir = Path(local_dir)

        # endswith() can check multiple suffixes if they are a tuple.
        lfs_files = tuple(lfs_files)
        futures = []

        with ThreadPoolExecutor(max_workers=50) as pool:
            for file_name in file_list:
                futures.append(pool.submit(self.download_executor, tag, file_name, lfs_files, local_dir.parent / 'temp'))

        try:
            _ = [i.result() for i in futures]
            move(local_dir.parent / 'temp', local_dir)
        except Exception as e:
            rmtree(local_dir.parent / 'temp')
            raise e

        if install_reqs:
            requirements = list(filter(lambda x: re.match(r'(.*/)?requirements.txt', x) is not None, file_list))
            for req in requirements:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', local_dir + req])

    def download(self, local_dir: Union[str, Path] = None, tag: str = 'main', install_reqs: bool = True):
        """
        Download repo without git.

        Agrs:
            local_dir (`Union[str, Path]`):
                Thre local dir to download the files into.
            tag (`str`):
                The tag of the repo to download.
            install_reqs (`bool`):
                Whether to install packages from requirements.txt.
        """
        if not self.exists():
            engine_log.error('%s/%s repo does not exist.', self._author, self._repo)
            raise ValueError(f'{self._author}/{self._author} repo does not exist.')

        if not local_dir:
            local_dir = Path.cwd()
        local_dir = Path(local_dir)

        lfs_files = self.obtain_lfs_extensions(tag)
        commit = self.latest_commit(tag)
        file_list = self.get_file_list(commit)
        self.download_files(tag=tag, file_list=file_list, lfs_files=lfs_files, local_dir=local_dir, install_reqs=install_reqs)
