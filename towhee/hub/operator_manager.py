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
