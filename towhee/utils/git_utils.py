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

import sys
import subprocess
import pkg_resources
from pathlib import Path
from typing import Union, List
from requests.exceptions import HTTPError
from shutil import rmtree
from pkg_resources import DistributionNotFound

from towhee.utils.hub_utils import HubUtils
from towhee.utils.log import engine_log


class GitUtils:
    """
    A wrapper class to wrap git manipulations.

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

        Raises:
            (`HTTPError`)
                Raise the error in request.
        """
        try:
            response = HubUtils(self._author, self._repo).get_info()
            return response.status_code == 200
        except HTTPError as e:
            raise e

    def clone(self, tag: str = 'main', install_reqs: bool = True, local_repo_path: Union[str, Path] = None) -> None:
        """
        Clone the repo to specified location.

        Args:
            tag (`str`):
                The tag name.
            install_reqs (`bool`):
                Whether to install packages from requirements.txt.
            self.local_repo_path (`Union[str, Path]`):
                The path to the local repo.

        Raises:
            (`ValueError`)
                Raise error if the repo does not exist.
        """
        if not local_repo_path:
            local_repo_path = Path.cwd() / self._repo_file_name
        local_repo_path = Path(local_repo_path)

        if not self.exists():
            engine_log.error('%s/%s repo does not exist.', self._author, self._repo)
            raise ValueError(f'{self._author}/{self._repo} repo does not exist.')

        url = f'{self._root}/{self._author}/{self._repo}.git'
        try:
            subprocess.check_call(['git', 'lfs'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            engine_log.warning(
                '\'git-lfs\' not found, execute download instead of clone. ' \
                'If you want to download large file with git-lfs, please install \'git-lfs\' and remove current local cache.'
            )
            raise FileNotFoundError('\'git-lfs\' not found, execute download instead of clone. ' \
                'If you want to download large file with git-lfs, please install \'git-lfs\' and remove current local cache.'
            ) from e

        try:
            print(f'Cloning the repo: {self._author}/{self._repo}... Be patient and waiting printing \'Successfully\'.')
            subprocess.check_call(['git', 'clone', '-b', tag, url, local_repo_path])
            print(f'Successfully clone the repo: {self._author}/{self._repo}.')
        except FileNotFoundError as e:
            engine_log.warning(
                '\'git\' not found, execute download instead of clone. ' \
                'If you want to check updates every time you run the pipeline, please install \'git\' and remove current local cache.'
            )
            raise FileNotFoundError(
                '\'git\' not found, execute download instead of clone. ' \
                'If you want to check updates every time you run the pipeline, please install \'git\' and remove current local cache.'
            ) from e
        except Exception as e: # pylint: disable=broad-except
            rmtree(local_repo_path)
            engine_log.error('Error when clone repo: %s/%s, will delete the local cache. Please check you network', self._author, self._repo)
            raise e

        if install_reqs and 'requirements.txt' in (i.name for i in local_repo_path.iterdir()):
            with open(local_repo_path / 'requirements.txt', 'r', encoding='utf-8') as f:
                reqs = f.read().split('\n')
            for req in reqs:
                try:
                    pkg_resources.require(req)
                except DistributionNotFound:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])

    def status(self):
        """
        Check if the local repo is out-dated.
        """
        try:
            status = subprocess.check_output(['git', 'rev-list', 'origin', '^HEAD'], stderr=subprocess.DEVNULL).decode('utf-8')
            return 'up-to-date' if not status else 'behind'
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError('The repo file has not .git.') from e

    def add(self, files: Union[str, List[str]] = None):
        """
        A wrapper function for git add.

        Stage current changes in the repo, please make sure your current woring directory is in the repo.

        Args:
            files (`Union[str, List[str]]`):
                The relative paths of the files to stage regard to the repo.
        """
        if not files:
            return subprocess.check_call(['git', 'add', '.'])

        if isinstance(files, str):
            return subprocess.check_call(['git', 'add', files])

        return subprocess.check_call(['git', 'add'] + files)

    def commit(self, cmt_msg: str):
        """
        A wrapper function for git commit.

        Commit current changes in the repo.

        Args:
            cmt_msg (`str`):
                The commit message.
        """
        return subprocess.check_call(['git', 'commit', '-sm', cmt_msg])

    def push(self, remote: str = 'origin', branch: str = 'main'):
        """
        A wrapper function for git push.

        Push local commits to remote, please make sure your current woring directory is in the repo.

        Args:
            self._repo_path (`Union[str, Path]`):
                The local repo cloned from remote.
            remote (`str`):
                The remote repo.
            branch (`str`):
                The remote branch.
        """
        return subprocess.check_call(['git', 'push', remote, branch])

    def pull(self, remote: str = 'origin', branch: str = 'main'):
        """
        A wrapper function for git pull.

        pull from remote, please make sure your current woring directory is in the repo.

        Args:
            self._repo_path (`Union[str, Path]`):
                The local repo cloned from remote.
            remote (`str`):
                The remote repo.
            branch (`str`):
                The remote branch.
        """
        try:
            res = subprocess.check_call(['git', 'pull', remote, branch])
        except subprocess.CalledProcessError as e:
            engine_log.error(e.output)
            raise e

        return res
