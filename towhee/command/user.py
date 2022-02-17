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

import random
import sys
import subprocess
from getpass import getpass
from requests.exceptions import HTTPError

from towhee.utils.hub_utils import HubUtils
from towhee.utils.hub_file_utils import HubFileUtils


class UserCommand:
    """
    Implementation for subcmd `towhee login`, `towhee whoami` and `towhee logout`.
    """
    def __init__(self, args) -> None:
        self._args = args
        self._root = 'https://towhee.io'
        self.hub = HubUtils()
        self.file = HubFileUtils()
        self._token = self.file.token

    def __call__(self) -> None:
        if self._args.action == 'login':
            self.login()
        elif self._args.action == 'whoami':
            username = self.whoami()
            if username:
                print(f'Username: {username}')
        elif self._args.action == 'logout':
            self.logout()

    @staticmethod
    def install(subparsers):
        subparsers.add_parser('login', help='user command: login using the same credentials as on towhee.io')

    def login(self) -> None:
        """
        Log in with https://towhee.io account.
        """
        if not self._token:
            username = input('Username: ')
            password = getpass()
            self.hub.set_author(username)
            try:
                r = self.hub.create_token(random.randint(0, 10000), password)
                token = r.json()['sha1']
            except HTTPError:
                print('Error password.')
                sys.exit()
            self.hub.login(password, token)
            self.write_to_credential_store(username, token)
            self.file.set_token(token)
            self.file.save()
            print('Successfully logged in.')
        else:
            print('You are already logged in, please log out first.')

    def whoami(self) -> str:
        if self._token:
            res = self.hub.get_user(self._token).json()
            username = res['username']
            return username
        else:
            print('Not logged it.')

    def logout(self) -> None:
        """
        Log out with https://towhee.io account.
        """
        if self._token:
            username = self.whoami()
            self.hub.set_author(username)
            self.hub.logout(self._token)
            self.erase_from_credential_store(username)
            self.file.delete()
            print('Done.')
        else:
            print('Not logged it.')

    def write_to_credential_store(self, username: str, password: str) -> None:
        """
        Write a token to store credentials.

        Args:
            username (`str`):
                authorized username.
            password (`str`):
                user password.
        """
        with subprocess.Popen(
            'git credential-store store'.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            input_username = f'username={username.lower()}'
            input_password = f'password={password}'

            process.stdin.write(
                f'url={self._root}\n{input_username}\n{input_password}\n\n'.encode('utf-8')
            )
            process.stdin.flush()

    def erase_from_credential_store(self, username=None) -> None:
        """
        Erases the credential store relative to https://towhee.io. If no `username` is specified, will erase the first entry.

        Args:
            username (`str`):
                authorized username.
        """
        with subprocess.Popen(
                'git credential-store erase'.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
        ) as process:
            standard_input = f'url={self._root}\n'

            if username is not None:
                standard_input += f'username={username.lower()}\n'

            standard_input += '\n'

            process.stdin.write(standard_input.encode('utf-8'))
            process.stdin.flush()


class LogoutCommand:
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        UserCommand(self._args)()

    @staticmethod
    def install(subparsers):
        subparsers.add_parser('logout', help='user command: logout')


class WhoCommand:
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        UserCommand(self._args)()

    @staticmethod
    def install(subparsers):
        subparsers.add_parser('whoami', help='user command: find out which towhee.io account you are logged in')
