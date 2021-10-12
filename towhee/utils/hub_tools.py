import requests
import os
import random
import sys
import getopt
from typing import List

from tempfile import TemporaryFile
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError


def create_token(user: str, password: str, token_name: str) -> str:
    """
    Creates an account verification token. This token allows for
    avoiding HttpBasicAuth for subsequent calls.

    Args:
        user: (`str`)
            The account name.
        password: (`str`)
            The account password.
        token_name: (`str`)
            The name to be given to the token.

    Returns:
        `int': The token id.
        `str`: The sha-1.

    Raises:
        HTTPError: Error in request.
    """

    url = f'https://hub.towhee.io/api/v1/users/{user}/tokens'
    data = {'name': token_name}
    try:
        r = requests.post(url, data=data, auth=HTTPBasicAuth(user, password))
        r.raise_for_status()
    except HTTPError as e:
        raise e

    res = r.json()
    token_id = str(res['id'])
    token_sha1 = str(res['sha1'])
    return token_id, token_sha1

def delete_token(user: str, password: str, token_id: int) -> None:
    """
    Deletes the token with the given name. Useful for cleanup after changes.

    Args:
        user: (`str`)
            The account name.
        password: (`str`)
            The account password.
        token_id(`int`)
            The token id.
    """
    url = f'https://hub.towhee.io/api/v1/users/{user}/tokens/{token_id}'
    try:
        r = requests.delete(url, auth=HTTPBasicAuth(user, password))
        r.raise_for_status()
    except HTTPError as e:
        raise e


def create_repo(repo: str, token: str, repo_type: str) -> None:
    """
    Creates a repo under the account connected to the passed in
    token.

    Args:
        repo: (`str`)
            Name of the repo to create.
        token: (`str`)
            Account verification token.
        repo_type: (`str`: 'model' | 'operator' | 'pipeline' | 'dataset')
            Which category of repo to create, only one can be used.

    Returns:
        None

    Raises:
        HTTPError: Error in request.
    """

    type_dict = {'model': 1, 'operator': 2, 'pipeline': 3, 'dataset': 4}

    # Commented out things in data that are breaking the creation
    data = {
        'auto_init': True,
        'default_branch': 'main',
        'description': 'This is another test repo',
        # 'gitignores': 'blah blah',
        # 'issue_labels': 'blah blah',
        # 'license': 'Blah Blah',
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


def delete_repo(user: str, repo: str, token: str) -> None:
    """
    Deletes the repo under the user, values correspond to
    https://www.hub.towhee/<user>/<repo>.

    Args:
        user: (`str`)
            The account name.
        repo: (`str`)
            The name of the repo to be deleted.
        token: (`str`)
            Account verification token for that user.

    Returns:
        None

    Raises:
        HTTPError: Error in request.
    """

    url = f'https://hub.towhee.io/api/v1/repos/{user}/{repo}'
    try:
        r = requests.delete(url, headers={'Authorization': 'token ' + token})
        r.raise_for_status()
    except HTTPError as e:
        raise e


def latest_branch_commit(user: str, repo: str, branch: str) -> str:
    """
    Grabs the latest commit for a specific branch.

    Args:
        user: (`str`)
            The account name.
        repo: (`str`)
            The repo name.
        branch: (`str`)
            The branch name.

    Returns:
        `str`: The branch commit hash cut down to 10 characters.

    Raises:
        HTTPError: Error in request.
    """

    url = f'https://hub.towhee.io/api/v1/repos/{user}/{repo}/commits?limit=1&page=1&sha={branch}'
    try:
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
    except HTTPError as e:
        raise e

    res = r.json()

    return res[0]['sha'][:10]



def obtain_lfs_extensions(user: str, repo: str, branch: str) -> List[str]:
    """
    Downloads the .gitattributes file from the specified repo
    in order to figure out which files are being tracked by git-lfs.

    Lines that deal with git-lfs take on the following format:

    ```
        *.extension   filter=lfs  merge=lfs ...
    ```

    Args:
        user: (`str`)
            The account name.
        repo: (`str`)
            The repo name.
        branch: (`str`)
            The branch name.

    Ret:
        `List[str]`: The list of file extentions tracked by git-lfs

    Raises:
        HTTPError: Error in request.

    """
    url = f'https://hub.towhee.io/api/v1/repos/{user}/{repo}/raw/.gitattributes?ref={branch}'
    lfs_files = []

    # Using temporary file in order to avoid double download, cleaner to not split up downloads everywhere.
    with TemporaryFile() as temp_file:
        try:
            r = requests.get(url)
            r.raise_for_status()
        except HTTPError as e:
            raise e

        temp_file.write(r.content)
        temp_file.seek(0)

        for line in temp_file:
            parts = line.split()
            if b'filter=lfs' in parts[1:]:  # only care if lfs filter is present
                lfs_files.append(parts[0].decode('utf-8')[1:])  # Removing the `*` in `*.ext`, need work if filtering specific files

    return lfs_files


def get_file_list(user: str, repo: str, commit: str) -> List[str]:
    """
    Gets all the files in the current repo at the given commit. This is done through forming a git tree
    recursively and filtering out all the files.

    Args:
        user: (`str`)
            The account name.
        repo: (`str`)
            The repo name.
        commit: (`str`)
            The commit to base current existing files.

    Returns:
        `List[str]`: The file paths for the repo

    Raises:
        HTTPError: Error in request.
    """

    url = f'https://hub.towhee.io/api/v1/repos/{user}/{repo}/git/trees/{commit}?recursive=1'
    file_list = []
    try:
        r = requests.get(url)
        r.raise_for_status()
    except HTTPError as e:
        raise e

    res = r.json()
    for file in res['tree']:  # Check each object in the tree
        if file['type'] != 'tree':  # Ignore directories (they have the type 'tree')
            file_list.append(file['path'])

    return file_list


def download_files(user: str, repo: str, branch: str, file_list: List[str], lfs_files: List[str], local_dir: str) -> None:
    """
    Download the files from hub. One url is used for git-lfs files and another for the other files.

    Args:
        user: (`str`)
            The account name.
        repo: (`str`)
            The repo name.
        branch: (`str`)
            The branch name.
        file_list: (`List[str]`)
            The hub file paths.
        lfs_files: (`List[str]`)
            The file extensions being tracked by git-lfs.
        local_dir: (`str`)
            The local directory to download to.

    Returns:
        None

    Raises:
        HTTPError: Error in request.
        OSError: Error in writing file.
    """
    if local_dir[-1] != '/': #  If the trailing forward slash is missing, add it on
        local_dir += '/'

    lfs_files = tuple(lfs_files)  # endswith() can check multiple suffixes if they are a tuple

    for file_path in file_list:
        if file_path.endswith(lfs_files):  # files dealt with lfs have a different url
            url = f'https://hub.towhee.io/{user}/{repo}/media/branch/{branch}/{file_path}'
            file_download_helper(url, local_dir + file_path)
        else:
            url = f'https://hub.towhee.io/api/v1/repos/{user}/{repo}/raw/{file_path}'
            file_download_helper(url, local_dir + file_path)


def file_download_helper(url, f) -> None:
    """
    Helper function that downloads using stream and writes files in chunks.
    """
    if not os.path.exists(os.path.dirname(f)):
        try:
            os.makedirs(os.path.dirname(f))
        except OSError as e:
            raise e
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
    except HTTPError as e:
        raise e

    with open(f, 'wb') as local_file:
        for chunk in r.iter_content(chunk_size=1024):
            local_file.write(chunk)


def download_repo(user: str, repo: str, branch: str, local_dir: str) -> None:
    """
    Performs a download of the selected repo to specified location.
    First checks to see if lfs is tracking files, then finds all the filepaths
    in the repo and lastly downloads them to the location.

    Args:
        user: (`str`)
            The account name.
        repo: (`str`)
            The repo name.
        branch: (`str`)
            The branch name.
        local_dir(`str`)
            The local directory being downloaded to

    Returns:
        None

    Raises:
        HTTPError: Error in request.
        OSError: Error in writing file.
    """
    lfs_files = obtain_lfs_extensions(user, repo, branch)
    commit = latest_branch_commit(user, repo, branch)
    file_list = get_file_list(user, repo, commit)
    download_files(user, repo, branch, file_list, lfs_files, local_dir)

def main(argv):
    try:
        opts, _ = getopt.getopt(argv[1:], 'u:p:r:t:b:d:',
                                ['create', 'delete', 'download', 'upload', 'user=', 'password=', 'repo=', 'type=', 'branch=', 'dir='])
    except getopt.GetoptError:
        print(
            'Usage: hub_interaction.py -<manipulate type> -u <user> -p ' +
                '<password> -r <repository> -t <repository type> -b <download branch> -d <download directory>'
        )
        sys.exit(2)
    else:
        if argv[0] not in ['create', 'delete', 'download', 'upload']:
            print('You must specify one kind of manipulation.')
            sys.exit(2)

    user = ''
    password = ''
    repo = ''
    repo_type = 'pipeline'
    branch = 'main'
    directory = os.getcwd() + '/test_download/'
    token_name = random.randint(0, 10000)  # going to have to figure out how to store the token
    manipulation = argv[0]

    for opt, arg in opts:
        if opt in ['-u', '--user']:
            user = arg
        elif opt in ['-p', '--password']:
            password = arg
        elif opt in ['-r', '--repo']:
            repo = arg
        elif opt in ['-t', '--type']:
            repo_type = arg
        elif opt in ['-d', '--dir']:
            directory = arg
        elif opt in ['-r', '--repo']:
            repo = arg

    if manipulation in ('create', 'delete'):
        if not user:
            user = input('Please enter your username: ')
        if not password:
            password = input('Please enter your password: ')
        if not repo:
            repo = input('Please enter the repo name: ')

        print('Creating token: ')
        token_id, token_hash = create_token(user, password, token_name)
        print('token: ', token_hash)

        if manipulation == 'create':
            print('Creating repo: ')
            create_repo(repo, token_hash, repo_type)

        elif manipulation == 'delete':
            print('Deleting repo: ')
            delete_repo(user, repo, token_hash)

        print('Deleting token: ')
        delete_token(user, password, token_id) # right now this doesnt get done if an exception is raised before it

    elif manipulation == 'download':
        if not user:
            user = input('Please enter the repo author: ')
        if not repo:
            repo = input('Please enter the repo name: ')
        print('Downloading repo:')
        download_repo(user, repo, branch, directory)


if __name__ == '__main__':
    main(sys.argv[1:])
