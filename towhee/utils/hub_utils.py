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
from requests.exceptions import HTTPError
from requests.auth import HTTPBasicAuth


class HubUtils:
    """
    The Hub Utils to deal with the http request. And other staticmethod for hub.

    Args:
        author (`str`):
            The author of the repo.
        repo (`str`):
            The name of the repo.
        root (`str`):
            The root url where the repo located.
    """
    def __init__(self, author: str = None, repo: str = None, root: str = 'https://towhee.io'):
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

    def set_author(self, author: str):
        self._author = author

    def set_repo(self, repo: str):
        self._repo = repo

    def get_info(self) -> requests.Response:
        """
        Get a repo.

        Returns:
            (`Response`)
                The information about the repo.

        Raises:
            (`HTTPError`)
                Raise the error in request.
        """
        url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}'
        response = requests.get(url)
        return response

    def get_commits(self, limit: int, page: int, tag: str) -> requests.Response:
        """
        Get a list of all commits of a repo.

        Args:
            limit (`int`):
                Page size of results.
            page (`int`):
                Page number of results.
            tag (`str`):
                The tag name.

        Returns:
            (`Response`)
                A list of all commits of a repo.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """

        url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}/commits?limit={limit}&page={page}&sha={tag}'
        try:
            response = requests.get(url, allow_redirects=True)
            response.raise_for_status()
            return response
        except HTTPError as e:
            raise e

    def get_tree(self, commit: str) -> requests.Response:
        """
        Get a git tree in the current repo at the given commit.

        Args:
            commit (`str`):
                The commit to base current existing files.

        Returns:
            (`Response`)
                A git tree for the repo

        Raises:
            (`HTTPError`)
                Raise error in request.
        """

        url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}/git/trees/{commit}?recursive=1'
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except HTTPError as e:
            raise e

    def get_file(self, file_path: str, tag: str) -> requests.Response:
        """
        Get a file from a repo.

        Args:
            file_path (`str`):
                filepath of the file to get.
            tag (`str`):
                The tag name.

        Returns:
            (`Response`)
                A file content in the file_path.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        url = f'{self._root}/api/v1/repos/{self._author}/{self._repo}/raw/{file_path}?ref={tag}'
        response = requests.get(url, stream=True)
        return response

    def get_lfs(self, file_path: str, tag: str) -> requests.Response:
        """
        Get a file from a repo.

        Args:
            file_path (`str`):
                filepath of the file to get.
            tag (`str`):
                The tag name.

        Returns:
            (`Response`)
                A file content in the file_path.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        url = f'{self._root}/{self._author}/{self._repo}/media/branch/{tag}/{file_path}'
        response = requests.get(url, stream=True)
        return response

    def create_token(self, token_name: str, password: str) -> requests.Response:
        """
        Create an account verification token. This token allows for avoiding HttpBasicAuth for subsequent calls.

        Args:
            token_name (`str`):
                The name to be given to the token.
            password (`str`):
                The password of current author.

        Returns:
            (`Response`)
                Return the token(id, name, sha1, token_last_eight).

        Raises:
            (`HTTPError`)
                Raise the error in request.
        """
        url = f'{self._root}/api/v1/users/{self._author}/tokens'
        data = {'name': token_name}
        try:
            response = requests.post(url, data=data, auth=HTTPBasicAuth(self._author, password))
            response.raise_for_status()
            return response
        except HTTPError as e:
            raise e

    def delete_token(self, token_id: int, password: str) -> None:
        """
        Delete the token with the given name. Useful for cleanup after changes.

        Args:
            token_id (`int`):
                The token id.
            password (`str`):
                The password of current author.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        url = f'{self._root}/api/v1/users/{self._author}/tokens/{token_id}'
        try:
            response = requests.delete(url, auth=HTTPBasicAuth(self._author, password))
            response.raise_for_status()
        except HTTPError as e:
            raise e

    def create(self, password: str, repo_class: int) -> None:
        """
        Create a repo under current account.

        Args:
            password (`str`):
                Current author's password.
            repo_class (`int`):
                The repo class

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        token_name = random.randint(0, 10000)
        r = self.create_token(token_name, password)
        token = r.json()

        data = {
            'auto_init': True,
            'default_branch': 'main',
            'description': repo_class,
            'name': self._repo,
            'private': False,
            'template': False,
            'trust_model': 'default',
            'type': repo_class
        }
        url = self._root + '/api/v1/user/repos'
        try:
            r = requests.post(url, data=data, headers={'Authorization': 'token ' + str(token['sha1'])})
            r.raise_for_status()
        except HTTPError as e:
            raise e
        finally:
            self.delete_token(str(token['id']), password)

    def create_repo(self, token: str, repo_class: int) -> None:
        """
        Create a repo under current account with token.

        Args:
            token (`str`):
                Authenticated user's token.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """

        data = {
            'auto_init': True,
            'default_branch': 'main',
            'description': repo_class,
            'name': self._repo,
            'private': False,
            'template': False,
            'trust_model': 'default',
            'type': repo_class
        }
        url = self._root + '/api/v1/user/repos'
        try:
            r = requests.post(url, data=data, headers={'Authorization': 'token ' + str(token)})
            r.raise_for_status()
        except HTTPError as e:
            raise e

    def add_tag(self, password, tag, tag_num):
        """
        Create a repo under current account.

        Args:
            password (`str`):
                Current author's password.
            tag (`str`):
                The tag name to attach.
            tag_num (`int`):
                Which enum # for tag.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        token_name = random.randint(0, 10000)
        r = self.create_token(token_name, password)
        token = r.json()
        data = {
            'name': tag,
            'pretty_name': tag,
            'type': tag_num
        }
        url = self._root + '/api/v1/repos/' + self._author + '/' + self._repo + '/repotag'

        try:
            r = requests.post(url, data=data, headers={'Authorization': 'token ' + str(token['sha1'])})
            r.raise_for_status()
        except HTTPError as e:
            raise e
        finally:
            self.delete_token(str(token['id']), password)

    def get_user(self, token: str) -> requests.Response:
        """
        Get the user info with token.

        Args:
            token (`str`):
                Authenticated user's token.

        Returns:
            (`Response`)
                Return the user(id, username etc.).

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        url = self._root + '/api/v1/user'
        try:
            response = requests.get(url, params={'token': token})
            response.raise_for_status()
            return response
        except HTTPError as e:
            raise e

    def login(self, password, token):
        """
        Login with Authenticated user.

        Args:
            password (`str`):
                Authenticated user's password.
            token (`str`):
                Authenticated user's token.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        url = self._root + '/user/login'
        data = {
            'user_name': self._author,
            'password': password,
            '_csrf': token
        }
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
        except HTTPError as e:
            raise e

    def logout(self, token):
        """
        Logout with token.

        Raises:
            (`HTTPError`)
                Raise error in request.
        """
        url = self._root + '/user/logout'
        data = {'_csrf': token}
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
        except HTTPError as e:
            raise e
        pass

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

    @staticmethod
    def update_text(ori_str_list: list, tar_str_list: list, file: str, new_file: str) -> None:
        """
        Update the file from ori_str_list to tar_str_list and rename it with the new filename.

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
