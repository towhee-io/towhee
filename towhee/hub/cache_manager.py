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

import os
from pathlib import Path
import threading

from .downloader import download_operator, operator_tag_path

DEFAULT_CACHE_DIR = '~/.towhee'
ENV_TOWHEE_HOME = 'TOWHEE_HOME'
_HUB_ROOT = None

def set_local_dir(d):
    global _HUB_ROOT
    _HUB_ROOT = os.path.expanduser(d)


def get_local_dir():
    if _HUB_ROOT is not None:
        return _HUB_ROOT
    cache_root = os.path.expanduser(os.getenv(ENV_TOWHEE_HOME, DEFAULT_CACHE_DIR))
    return cache_root


class CacheManager:
    """
    Downloading from hub
    """
    def __init__(self):
        self._download_lock = threading.Lock()

    def _op_cache_name(self, author: str, repo: str, tag: str):
        if author == 'local':
            cache_root = os.environ.get('TEST_CACHE')
            return Path(cache_root) / repo.replace('-', '_')
        else:
            cache_root = get_local_dir()
            return operator_tag_path(Path(cache_root) / 'operators' / author / repo, tag)

    def get_operator(self, operator: str, tag: str, install_reqs: bool, latest: bool) -> Path:
        """Obtain the path to the requested operator.

        This function will obtain the first reference to the operator from the cache locations.
        If no opeartor is found, this function will download it to the default cache location
        from the Towhee hub.

        Args:
            operator (`str`):
                The operator in 'author/repo' format. Author will be 'local' if locally imported.
            tag (`str`):
                Which tag version to use of the opeartor. Will use 'main' if locally imported.
            install_reqs (`bool`):
                Whether to download the python packages if a requirements.txt file is included in the repo.
            latest (`bool`):
                Whether to download the latest files.

        Returns:
            (Path | None)
                Returns the path of the operator, None if a local operator isnt found.

        Raises:
            (`ValueError`):
                Incorrect opeartor format.
        """
        operator_split = operator.split('/')
        # For now assuming all piplines will be classifed as 'author/repo'.
        if len(operator_split) != 2:
            raise ValueError('''Incorrect operator format, should be '<author>/<operator_repo>'.''')
        author, repo = operator_split
        op_path = self._op_cache_name(author, repo, tag)

        if op_path.is_dir() and not latest:
            return op_path

        with self._download_lock:
            download_path = op_path.parent.parent
            download_operator(author, repo, tag, download_path, install_reqs, latest)
            return op_path
