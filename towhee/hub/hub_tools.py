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
import os
import random
import sys
import getopt
import time
import subprocess
import re
import yaml
from pathlib import Path
from importlib import import_module

from typing import List, Tuple
from tqdm import tqdm
from threading import Thread
from getpass import getpass
import git

from tempfile import TemporaryFile
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError

### Repo Download related functions


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


def exists(author: str, repo: str) -> bool:
    """
    Check if a repo exists.

    Args:
        author (`str`):
            The author name.
        repo (`str`):
            The repo name.

    Returns:
        (`bool`)
            return `True` if the repository exists, else `False`.
    """
    try:
        url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}'
        r = requests.get(url)
        return r.status_code == 200
    except HTTPError as e:
        raise e


def obtain_lfs_extensions(author: str, repo: str, tag: str) -> List[str]:
    """
    Download the .gitattributes file from the specified repo in order to figure out
    which files are being tracked by git-lfs.

    Lines that deal with git-lfs take on the following format:

    ```
        *.extension   filter=lfs  merge=lfs ...
    ```

    Args:
        author (`str`):
            The account name.
        repo (`str`):
            The repo name.
        tag (`str`):
            The tag name.

    Returns:
        (`List[str]`)
            The list of file extentions tracked by git-lfs.
    """
    url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}/raw/.gitattributes?ref={tag}'
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


def latest_branch_commit(author: str, repo: str, branch: str) -> str:
    """
    Grab the latest commit for a specific branch.

    Args:
        author (`str`):
            The account name.
        repo (`str`):
            The repo name.
        branch (`str`):
            The branch name.

    Returns:
        (`str`)
            The branch commit hash cut down to 10 characters.

    Raises:
        (`HTTPError`)
            Raise error in request.
    """

    url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}/commits?limit=1&page=1&sha={branch}'
    try:
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
    except HTTPError as e:
        raise e

    res = r.json()

    return res[0]['sha'][:10]


def get_file_list(author: str, repo: str, commit: str) -> List[str]:
    """
    Get all the files in the current repo at the given commit.

    This is done through forming a git tree recursively and filtering out all the files.

    Args:
        author (`str`):
            The account name.
        repo (`str`):
            The repo name.
        commit (`str`):
            The commit to base current existing files.

    Returns:
        (`List[str]`)
            The file paths for the repo

    Raises:
        (`HTTPError`)
            Raise error in request.
    """

    url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}/git/trees/{commit}?recursive=1'
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


def download_files(author: str, repo: str, tag: str, file_list: List[str], lfs_files: List[str], local_dir: str, install_reqs: bool) -> None:
    """
    Download the files from hub. One url is used for git-lfs files and another for the other files.

    Args:
        author (`str`):
            The account name.
        repo (`str`):
            The repo name.
        tag (`str`):
            The tag name.
        file_list (`List[str]`):
            The hub file paths.
        lfs_files (`List[str]`):
            The file extensions being tracked by git-lfs.
        local_dir (`str`):
            The local directory to download to.
        install_reqs (`bool`):
            Whether to install packages from requirements.txt

    Raises:
        (`HTTPError`)
            Rasie error in request.
        (`OSError`)
            Raise error in writing file.
    """
    threads = []

    # If the trailing forward slash is missing, add it on.
    if local_dir[-1] != '/':
        local_dir += '/'

    # endswith() can check multiple suffixes if they are a tuple.
    lfs_files = tuple(lfs_files)

    for file_name in file_list:
        # Files dealt with lfs have a different url.
        if file_name.endswith(lfs_files):
            url = f'https://hub.towhee.io/{author}/{repo}/media/branch/{tag}/{file_name}'
        else:
            url = f'https://hub.towhee.io/api/v1/repos/{author}/{repo}/raw/{file_name}?ref={tag}'

        threads.append(Worker(url, local_dir, file_name))
        threads[-1].start()

    for thread in threads:
        thread.join()

    if install_reqs:
        requirements = list(filter(lambda x: re.match(r'(.*/)?requirements.txt', x) is not None, file_list))
        for req in requirements:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', local_dir + req])


def download_repo(author: str, repo: str, tag: str, local_dir: str, install_reqs: bool = True) -> None:
    """
    Performs a download of the selected repo to specified location.

    First checks to see if lfs is tracking files, then finds all the filepaths
    in the repo and lastly downloads them to the location.

    Args:
        author (`str`):
            The account name.
        repo (`str`):
            The repo name.
        tag (`str`):
            The tag name.
        local_dir (`str`):
            The local directory being downloaded to
        install_reqs (`bool`):
            Whether to install packages from requirements.txt

    Raises:
        (`HTTPError`)
            Raise error in request.
        (`OSError`)
            Raise error in writing file.
    """
    if not exists(author, repo):
        raise ValueError(author + '/' + repo + ' repo doesnt exist.')

    # lfs_files = obtain_lfs_extensions(author, repo, tag)
    # commit = latest_branch_commit(author, repo, tag)
    # file_list = get_file_list(author, repo, commit)
    # download_files(author, repo, tag, file_list, lfs_files, local_dir, install_reqs)
    url = f'https://towhee.io/{author}/{repo}.git'
    git.Repo.clone_from(url=url, to_path=local_dir, branch=tag)

    if install_reqs:
        if 'requirements.txt' in os.listdir(local_dir):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', Path(local_dir) / 'requirements.txt'])


### Repo Creation related functions


def create_token(author: str, password: str, token_name: str) -> Tuple[int, str]:
    """
    Create an account verification token.

    This token allows for avoiding HttpBasicAuth for subsequent calls.

    Args:
        author (`str`):
            The account name.
        password (`str`):
            The account password.
        token_name (`str`):
            The name to be given to the token.

    Returns:
        (`Tuple[int, str]`)
            Return the token id and the sha-1.

    Raises:
        (`HTTPError`)
            Raise the error in request.
    """
    url = f'https://hub.towhee.io/api/v1/users/{author}/tokens'
    data = {'name': token_name}
    try:
        r = requests.post(url, data=data, auth=HTTPBasicAuth(author, password))
        r.raise_for_status()
    except HTTPError as e:
        raise e

    res = r.json()
    token_id = str(res['id'])
    token_sha1 = str(res['sha1'])

    return token_id, token_sha1


def delete_token(author: str, password: str, token_id: int) -> None:
    """
    Delete the token with the given name. Useful for cleanup after changes.

    Args:
        author (`str`);
            The account name.
        password (`str`):
            The account password.
        token_id (`int`):
            The token id.
    """
    url = f'https://hub.towhee.io/api/v1/users/{author}/tokens/{token_id}'
    try:
        r = requests.delete(url, auth=HTTPBasicAuth(author, password))
        r.raise_for_status()
    except HTTPError as e:
        raise e


def create_repo(repo: str, token: str, repo_type: str) -> None:
    """
    Create a repo under the account connected to the passed in token.

    Args:
        repo (`str`):
            Name of the repo to create.
        token (`str`):
            Account verification token.
        repo_type (`str`):
            Which category of repo to create, only one can be used, includes
            ('model', 'operator', 'pipeline', 'dataset').

    Raises:
        (`HTTPError`)
            Raise error in request.
    """

    type_dict = {'model': 1, 'operator': 2, 'pipeline': 3, 'dataset': 4}

    # Commented out things in data that are breaking the creation
    data = {
        'auto_init': True,
        'default_tag': 'main',
        'description': 'This is another test repo',
        'name': repo,
        'private': False,
        'template': False,
        'trust_model': 'default',
        'type': type_dict[repo_type]
    }
    url = 'https://hub.towhee.io/api/v1/user/repos'
    try:
        r = requests.post(url, data=data, headers={'Authorization': 'token ' + token})
        r.raise_for_status()
    except HTTPError as e:
        raise e


def init_file_structure(author: str, repo: str, repo_type: str) -> None:
    """
    Initialized the file structure with template.

    First clone the repo, then download and rename the template repo file.

    Args:
        author (`str`):
            The account name.
        repo (`str`):
            The repo name.
        repo_type (`str`):
            Which category of repo to create, only one can be used, includes
            ('model', 'operator', 'pipeline', 'dataset').

    Raises:
        (`HTTPError`)
            Raise error in request.
        (`OSError`)
            Raise error in writing file.
    """
    links = 'https://hub.towhee.io/' + author + '/' + repo + '.git'
    subprocess.call(['git', 'clone', links])
    repo_file_name = repo.replace('-', '_')
    if repo_type == 'operator':
        # TODO: distinguish nnop and pyop (Shiyu)
        lfs_files = obtain_lfs_extensions('towhee', 'operator-template', 'main')
        commit = latest_branch_commit('towhee', 'operator-template', 'main')
        file_list = get_file_list('towhee', 'operator-template', commit)
        download_files('towhee', 'operator-template', 'main', file_list, lfs_files, str(Path.cwd() / repo), False)

        (Path(repo) / 'operator_template.py').rename(Path(repo) / (repo_file_name + '.py'))
        (Path(repo) / 'operator_template.yaml').rename(Path(repo) / (repo_file_name + '.yaml'))

    elif repo_type == 'pipeline':
        lfs_files = obtain_lfs_extensions('towhee', 'pipeline-template', 'main')
        commit = latest_branch_commit('towhee', 'pipeline-template', 'main')
        file_list = get_file_list('towhee', 'pipeline-template', commit)
        download_files('towhee', 'pipeline-template', 'main', file_list, lfs_files, str(Path.cwd() / repo), False)

        (Path(repo) / 'pipeline_template.yaml').rename(Path(repo) / (repo_file_name + '.yaml'))


def covert_dic(dicts: dict) -> dict:
    """
    Convert all the values in a dictionary to str and replace char, for example:
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
    return dicts


def generate_repo_yaml(author: str, repo: str) -> None:
    """
    Generate the yaml of Operator, for example:
    name: 'operator-template'
    labels:
      recommended_framework: ''
      class: ''
      others: ''
    operator: 'towhee/operator-template'
    init:
      model_name: str
    call:
      input:
        img_path: str
      output:
        feature_vector: numpy.ndarray

    Args:
        author (`str`):
            The account name.
        repo (`str`):
            The repo name.

    Raises:
        (`HTTPError`)
            Raise error in request.
        (`OSError`)
            Raise error in writing file.
    """
    sys.path.append(repo)
    repo_file_name = repo.replace('-', '_')
    # get class name in camel case
    components = repo.split('-')
    class_name = ''.join(x.title() for x in components)
    yaml_file = repo + '/' + repo_file_name + '.yaml'
    operator_name = author + '/' + repo
    # import the class from repo
    cls = getattr(import_module(repo_file_name, repo), class_name)

    init_args = cls.__init__.__annotations__
    try:
        del init_args['return']
    except KeyError:
        pass

    call_func = cls.__call__.__annotations__
    try:
        call_output = call_func.pop('return')
        call_output = call_output.__annotations__
    except KeyError:
        pass
    call_input = call_func

    data = {
        'name': repo,
        'labels': {
            'recommended_framework': '', 'class': '', 'others': ''
        },
        'operator': operator_name,
        'init': covert_dic(init_args),
        'call': {
            'input': covert_dic(call_input), 'output': covert_dic(call_output)
        }
    }
    with open(yaml_file, 'w', encoding='utf-8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)


def main(argv):
    try:
        opts, _ = getopt.getopt(
            argv[1:],
            'a:p:r:t:b:d:',
            ['create', 'download', 'generate-yaml', 'init', 'author=', 'password=', 'repo=', 'type=', 'tag=', 'dir=']
        )
    except getopt.GetoptError:
        print(
            'Usage: hub_tool.py -<manipulate type> -a <author> -p ' +
            '<password> -r <repository> -t <repository type> -b <download tag> -d <download directory>'
        )
        sys.exit(2)
    else:
        if argv[0] not in ['create', 'download', 'init', 'generate-yaml']:
            print('You must specify one kind of manipulation.')
            sys.exit(2)

    author = ''
    password = ''
    repo = ''
    repo_type = ''
    tag = 'main'
    directory = Path.cwd()
    # TODO(Filip) figure out how to store the token
    token_name = random.randint(0, 10000)
    manipulation = argv[0]

    for opt, arg in opts:
        if opt in ['-a', '--author']:
            author = arg
        elif opt in ['-p', '--password']:
            password = arg
        elif opt in ['-r', '--repo']:
            repo = arg
        elif opt in ['-t', '--type']:
            repo_type = arg
        elif opt in ['-b', '--tag']:
            tag = arg
        elif opt in ['-d', '--dir']:
            directory = Path(arg)

    if manipulation == 'create':
        if not author:
            author = input('Please enter your username: ')
        if not password:
            password = getpass('Please enter your password: ')
        if not repo:
            repo = input('Please enter the repo name: ')
        if not repo_type:
            repo_type = input('Please enter the repo type, choose one from "operator | pipeline": ')

        print('Creating token...')
        token_id, token_hash = create_token(author, password, token_name)
        print('token: ', token_hash)

        print('Creating repo...')
        create_repo(repo, token_hash, repo_type)
        init_choice = input('Do you want to clone the repo from hub with template? [Y|n]\n')

        if init_choice.lower() in ['yes', 'y']:
            print('Clone with template...')
            init_file_structure(author, repo, repo_type)

        print('Done')

        print('Deleting token...')
        # TODO(Filip) right now this doesnt get done if an exception is raised before it
        delete_token(author, password, token_id)
        print('Done')

    elif manipulation == 'download':
        if not author:
            author = input('Please enter the repo author: ')
        if not repo:
            repo = input('Please enter the repo name: ')
        if not repo_type:
            repo = input('Please enter the repo type: ')
        print('Downloading repo...')
        directory = directory / (repo_type + 's') / author / repo / tag
        download_repo(author, repo, tag, directory)

    elif manipulation == 'generate-yaml':
        if not author:
            author = input('Please enter the repo author: ')
        if not repo:
            repo = input('Please enter the operator repo name: ')
        print('Generating the yaml of repo...')
        generate_repo_yaml(author, repo)
        print('Done')

    elif manipulation == 'init':
        if not author:
            author = input('Please enter the repo author: ')
        if not repo:
            repo = input('Please enter the repo name: ')
        if not repo_type:
            repo_type = input('Please enter the repo type, choose one from "operator | pipeline": ')
        print('Clone with template...')
        init_file_structure(author, repo, repo_type)


if __name__ == '__main__':
    main(sys.argv[1:])
