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

from pathlib import Path


class HubFileUtils:
    """
    Class for towhee file about hub token.
    """
    def __init__(self):
        self._file = (Path.home() / '.towhee/token').resolve()
        self._token = self.get_token()

    @property
    def token(self):
        return self._token

    @property
    def file(self):
        return self._file

    def set_token(self, token) -> None:
        self._token = token

    def get_token(self) -> str:
        try:
            return self._file.read_text()
        except FileNotFoundError:
            pass

    def save(self) -> None:
        if not self._file.parent.exists():
            self._file.parent.mkdir(parents=True)
        self._file.write_text(self._token)

    def delete(self) -> None:
        self._file.unlink()
