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

import re
import subprocess
import sys
import time
from typing import Tuple, List, Union
from shutil import rmtree
from pathlib import Path
from requests.exceptions import HTTPError

from tqdm import tqdm
from tempfile import TemporaryFile
from concurrent.futures import ThreadPoolExecutor
from towhee.utils.hub_utils import HubUtils
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
    def __init__(self, author: str, repo: str, root: str = 'https://towhee.io'):
        self._author = author
        self._repo = repo
        self._root = root
        self._temp = {'pipeline': 'pipeline-template', 'pyoperator': 'pyoperator-template', 'nnoperator': 'nnoperator-template'}
        self.hub_utils = HubUtils(self._author, self._repo, self._root)

    @property
    def author(self):
        return self._author

    @property
    def repo(self):
        return self._repo

    @property
    def root(self):
        return self._root

    def exists(self) -> bool:
        """
        Check if a repo exists.

        Returns:
            (`bool`)
                return `True` if the repository exists, else `False`.
        """
        res = self.hub_utils.get_info()
        return res.status_code == 200

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
        res = self.hub_utils.get_file('.gitattributes', tag)
        lfs_files = []
        if res.status_code == 200:
            # Using temporary file in order to avoid double download, cleaner to not split up downloads everywhere.
            with TemporaryFile() as temp_file:
                temp_file.write(res.content)
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
        res = self.hub_utils.get_commits(limit=1, page=1, tag=tag)
        return res.json()[0]['sha'][:10]

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
        file_list = []
        res = self.hub_utils.get_tree(commit)

        # Check each object in the tree
        for file in res.json()['tree']:
            # Ignore directories (they have the type 'tree')
            if file['type'] != 'tree':
                file_list.append(file['path'])
        return file_list

    def download_executor(self, tag: str, file_name: str, lfs_files: Tuple[str], local_repo_path: Union[str, Path]) -> None:
        """
        Load the content from url and write into local files.

        Args:
            tag (`str`):
                The tag name.
            file_name (`str`):
                The hub file paths.
            lfs_files (`str):
                The file extensions being tracked by git-lfs.
            local_repo_path (`Union[str, Path]`):
                The local directory to download to.

        Raises:
            (`HTTPError`)
                Rasie error in request.
            (`OSError`)
                Raise error in writing file.
        """
        file_path = Path(local_repo_path) / file_name
        if not file_path.parent.resolve().exists():
            try:
                file_path.parent.resolve().mkdir(parents=True)
            except FileExistsError:
                pass
            except OSError as e:
                raise e

        # Get content.
        for i in range(5):
            try:
                if file_name.endswith(lfs_files):
                    res = self.hub_utils.get_lfs(file_name, tag)
                else:
                    res = self.hub_utils.get_file(file_name, tag)

                if res.status_code == 200:
                    break
                elif res.status_code == 429 and i < 4:
                    time.sleep(2)
                    continue
                res.raise_for_status()
            except HTTPError as e:
                raise e

        # Create local files.
        file_size = int(res.headers.get('content-length', 0))
        chunk_size = 1024
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f'Downloading {file_name}')
        with open(file_path, 'wb') as local_file:
            for chunk in res.iter_content(chunk_size=chunk_size):
                local_file.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()

    def download_files(self, tag: str, file_list: List[str], lfs_files: List[str], local_repo_path: Union[str, Path], install_reqs: bool) -> None:
        """
        Download the given files.

        Agrs:
            tag (`str`):
                The tag of the repo to download.
            file_list (List[str]):
                The files to download
            lfs_files (List[str]):
                The lfs files extensions.
            local_repo_path (`Union[str, Path]`):
                Thre local dir to download the files into.
            install_reqs (`bool`):
                Whether to install packages from requirements.txt.
        """
        local_repo_path = Path(local_repo_path)

        # endswith() can check multiple suffixes if they are a tuple.
        lfs_files = tuple(lfs_files)
        futures = []

        temp_path = local_repo_path.parent / 'dl-temp'
        with ThreadPoolExecutor(max_workers=50) as pool:
            for file_name in file_list:
                futures.append(pool.submit(self.download_executor, tag, file_name, lfs_files, temp_path))

        try:
            _ = [i.result() for i in futures]
            temp_path.rename(local_repo_path)
        except Exception as e:
            rmtree(temp_path)
            raise e

        if install_reqs:
            requirements = list(filter(lambda x: re.match(r'(.*/)?requirements.txt', x) is not None, file_list))
            for req in requirements:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(local_repo_path / req)])

    def download(self, local_repo_path: Union[str, Path] = Path.cwd(), tag: str = 'main', install_reqs: bool = True) -> None:
        """
        Download repo without git.

        Agrs:
            local_repo_path (`Union[str, Path]`):
                Thre local dir to download the files into.
            tag (`str`):
                The tag of the repo to download.
            install_reqs (`bool`):
                Whether to install packages from requirements.txt.
        """
        if not self.exists():
            engine_log.error('%s/%s repo does not exist.', self._author, self._repo)
            raise ValueError(f'{self._author}/{self._author} repo does not exist.')

        local_repo_path = Path(local_repo_path)
        lfs_files = self.obtain_lfs_extensions(tag)
        commit = self.latest_commit(tag)
        file_list = self.get_file_list(commit)
        self.download_files(tag=tag, file_list=file_list, lfs_files=lfs_files, local_repo_path=local_repo_path, install_reqs=install_reqs)
