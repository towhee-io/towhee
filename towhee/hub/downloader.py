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
import sys
import tempfile
import shutil
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import requests
from tqdm import tqdm

from towhee.utils.hub_utils import HubUtils


ENV_TOWHEE_URL = 'TOWHEE_URL'
_HUB_URL = 'https://towhee.io'


def set_hub_url(url):
    global _HUB_URL
    _HUB_URL = url


def get_hub_url():
    return os.getenv(ENV_TOWHEE_URL, _HUB_URL)


class _HubFiles:
    '''
    meta_infos:
    [
      {
        "path": "README.md",
        "mode": "100644",
        "type": "blob",
        "size": 5310,
        "sha": "133cd84e7a4bf39b49ffb18f3f2628afe93dd5b4",
        "downloadLink": "https://towhee.io/api/v1/repos/image-embedding/isc/git/blobs/133cd84e7a4bf39b49ffb18f3f2628afe93dd5b4"
      },
      {
        "path": "benchmark",
        "mode": "040000",
        "type": "tree",
        "size": 0,
        "sha": "65c110ada6d3fb4e477cc57b67026e37a4499e39",
        "downloadLink": "https://towhee.io/api/v1/repos/image-embedding/isc/git/trees/65c110ada6d3fb4e477cc57b67026e37a4499e39"
      },
      {
        "path": "benchmark/qps_test.py",
        "mode": "100644",
        "type": "blob",
        "size": 3617,
        "sha": "2bc4af18560f0597d5efd037aef7111aba01d140",
        "downloadLink": "https://towhee.io/api/v1/repos/image-embedding/isc/git/blobs/2bc4af18560f0597d5efd037aef7111aba01d140"
      }
    ]
    '''
    def __init__(self, root: str, tag: str, meta_infos=None):
        self._root = Path(root).absolute()
        self._tag = tag.replace('-', '_').replace('.', '_')
        self._meta_infos = meta_infos

    @property
    def file_path(self):
        files_dir = self._root / 'files'
        files_dir.mkdir(parents=True, exist_ok=True)
        return files_dir

    def get_tag_path(self, create = False):
        versions_dir = self._root / 'versions' / self._tag
        if create:
            versions_dir.mkdir(parents=True, exist_ok=True)
        return versions_dir

    @property
    def requirements(self):
        for item in self._meta_infos:
            if item['type'] == 'blob' and item['path'] == 'requirements.txt':
                return self.get_tag_path() / item['path']
        return None

    def symlink_files(self):
        tmp_dir = Path(tempfile.mkdtemp(dir=self._root))
        for dst, src in self.symlink_pair():
            dst_file = tmp_dir / dst
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.symlink_to(src)
        tmp_dir.rename(self.get_tag_path(True))

    def symlink_pair(self):
        pair = []
        for item in self._meta_infos:
            if item['type'] == 'tree':
                continue
            pair.append((Path(item['path']), self.file_path / item['sha']))
        return pair

    def local_url_pair(self):
        pair = []
        for item in self._meta_infos:
            if item['type'] == 'tree':
                continue
            pair.append((self.file_path / item['sha'], item['downloadLink']))
        return pair


class _Downloader:
    """
    Download op
    """
    def __init__(self, hub_files: 'HubFiles'):
        self._hub_files = hub_files

    @staticmethod
    def download_url_to_file(url: str, dst: str, file_size: int = None) -> bool:
        """Download file at the given URL to a local path.
        Args:
            url (str): URL of the file to download
            dst (str): Full path where file will be saved
        """

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            try:
                if file_size is None:
                    file_size = int(r.headers.get('content-length', 0))
                dst_dir = os.path.dirname(dst)
                with tempfile.NamedTemporaryFile(delete=False, dir=dst_dir) as f:
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Downloading {url} to {dst_dir}') as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            real_size = f.write(chunk)
                            pbar.update(real_size)
                shutil.move(f.name, dst)
            finally:
                if os.path.exists(f.name):
                    os.remove(f.name)

    def download(self):
        futures = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            for local_file, url in self._hub_files.local_url_pair():
                if not local_file.is_file():
                    futures.append(pool.submit(_Downloader.download_url_to_file, url, local_file))
            _ = [i.result() for i in futures]


_DOWNLOAD_LOCK = threading.Lock()

def download_operator(author: str, repo: str, tag: str, op_path: Path, install_reqs: bool = True):
    hub_url = get_hub_url()
    ht = HubUtils(author, repo, hub_url)
    meta = ht.branch_tree(tag)
    if meta is None:
        raise RuntimeError('Fetch op {}/{}:{} info failed'.format(author, repo, tag))
    fs = _HubFiles(op_path, tag, meta)
    _Downloader(fs).download()
    fs.symlink_files()

    if install_reqs and fs.requirements:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', fs.requirements])


def operator_tag_path(root: Path, tag: str):
    return _HubFiles(root, tag).get_tag_path()
