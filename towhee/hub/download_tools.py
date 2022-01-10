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
import time
from pathlib import Path

from tqdm import tqdm
from threading import Thread
from requests.exceptions import HTTPError


class Worker(Thread):
    """
    Worker class to realize multi-threads download.

    Args:
        url (`str`):
            The url of the target files.
        local_dir (`str`):
            The local directory to download to.
        file_name (`str`):
            The name of the file (includes extension).
    """
    def __init__(self, url: str, local_dir: str, file_name: str):
        super().__init__()
        self.url = url
        self.local_dir = local_dir
        self.file_name = file_name

    def run(self):
        # Creating the directory tree to the file.
        file_path = self.local_dir + self.file_name
        if not Path(file_path).parent.resolve().exists():
            try:
                Path(file_path).parent.resolve().mkdir()
            except FileExistsError:
                pass
            except OSError as e:
                raise e

        # Get content.
        try:
            r = requests.get(self.url, stream=True)
            if r.status_code == 429:
                time.sleep(3)
                self.run()
            r.raise_for_status()
        except HTTPError as e:
            raise e

        # Create local files.
        file_size = int(r.headers.get('content-length', 0))
        chunk_size = 1024
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f'Downloading {self.file_name}')
        with open(self.local_dir + self.file_name, 'wb') as local_file:
            for chunk in r.iter_content(chunk_size=chunk_size):
                local_file.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()


# def obtain_lfs_extensions(author: str, repo: str, tag: str) -> List[str]:
#     """
#     Download the .gitattributes file from the specified repo in order to figure out
#     which files are being tracked by git-lfs.

#     Lines that deal with git-lfs take on the following format:

#     ```
#         *.extension   filter=lfs  merge=lfs ...
#     ```

#     Args:
#         author (`str`):
#             The account name.
#         repo (`str`):
#             The repo name.
#         tag (`str`):
#             The tag name.

#     Returns:
#         (`List[str]`)
#             The list of file extentions tracked by git-lfs.
#     """
#     url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}/raw/.gitattributes?ref={tag}'
#     lfs_files = []

#     # Using temporary file in order to avoid double download, cleaner to not split up downloads everywhere.
#     with TemporaryFile() as temp_file:
#         try:
#             r = requests.get(url)
#             r.raise_for_status()
#         except HTTPError:
#             return lfs_files

#         temp_file.write(r.content)
#         temp_file.seek(0)

#         for line in temp_file:
#             parts = line.split()
#             # We only care if lfs filter is present.
#             if b'filter=lfs' in parts[1:]:
#                 # Removing the `*` in `*.ext`, need work if filtering specific files.
#                 lfs_files.append(parts[0].decode('utf-8')[1:])

#     return lfs_files

# def latest_commit(author: str, repo: str, tag: str) -> str:
#     """
#     Grab the latest commit of a tag.

#     Args:
#         author (`str`):
#             The account name.
#         repo (`str`):
#             The repo name.
#         tag (`str`):
#             The tag name.

#     Returns:
#         (`str`)
#             The latest commit hash cut down to 10 characters.

#     Raises:
#         (`HTTPError`)
#             Raise error in request.
#     """

#     url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}/commits?limit=1&page=1&sha={tag}'
#     try:
#         r = requests.get(url, allow_redirects=True)
#         r.raise_for_status()
#     except HTTPError as e:
#         raise e

#     res = r.json()

#     return res[0]['sha'][:10]

# def get_file_list(author: str, repo: str, commit: str) -> List[str]:
#     """
#     Get all the files in the current repo at the given commit.

#     This is done through forming a git tree recursively and filtering out all the files.

#     Args:
#         author (`str`):
#             The account name.
#         repo (`str`):
#             The repo name.
#         commit (`str`):
#             The commit to base current existing files.

#     Returns:
#         (`List[str]`)
#             The file paths for the repo

#     Raises:
#         (`HTTPError`)
#             Raise error in request.
#     """

#     url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}/git/trees/{commit}?recursive=1'
#     file_list = []
#     try:
#         r = requests.get(url)
#         r.raise_for_status()
#     except HTTPError as e:
#         raise e

#     res = r.json()
#     # Check each object in the tree
#     for file in res['tree']:
#         # Ignore directories (they have the type 'tree')
#         if file['type'] != 'tree':
#             file_list.append(file['path'])

#     return file_list

# def download_files(author: str, repo: str, tag: str, file_list: List[str], lfs_files: List[str], local_dir: str, install_reqs: bool) -> None:
#     """
#     Download the files from hub. One url is used for git-lfs files and another for the other files.

#     Args:
#         author (`str`):
#             The account name.
#         repo (`str`):
#             The repo name.
#         tag (`str`):
#             The tag name.
#         file_list (`List[str]`):
#             The hub file paths.
#         lfs_files (`List[str]`):
#             The file extensions being tracked by git-lfs.
#         local_dir (`str`):
#             The local directory to download to.
#         install_reqs (`bool`):
#             Whether to install packages from requirements.txt

#     Raises:
#         (`HTTPError`)
#             Rasie error in request.
#         (`OSError`)
#             Raise error in writing file.
#     """
#     threads = []

#     # If the trailing forward slash is missing, add it on.
#     if local_dir[-1] != '/':
#         local_dir += '/'

#     # endswith() can check multiple suffixes if they are a tuple.
#     lfs_files = tuple(lfs_files)

#     for file_name in file_list:
#         # Files dealt with lfs have a different url.
#         if file_name.endswith(lfs_files):
#             url = f'https://hub.towhee.io/{author}/{repo}/media/branch/{tag}/{file_name}'
#         else:
#             url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}/raw/{file_name}?ref={tag}'

#         threads.append(Worker(url, local_dir, file_name))
#         threads[-1].start()

#     for thread in threads:
#         thread.join()

#     if install_reqs:
#         requirements = list(filter(lambda x: re.match(r'(.*/)?requirements.txt', x) is not None, file_list))
#         for req in requirements:
#             subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', local_dir + req])
