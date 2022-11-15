# Copyright 2022 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
from urllib import request

import warnings
from tqdm import tqdm


def download_from_url(url: str, root: str = None, hash_prefix: str = None):
    """
    Download file from url.

    Args:
        url (`str`): url
        root (`str`): root directory to save downloaded file, defaults to '~/.towhee/checkpoints'
        hash_prefix (`str`): hash prefix to checksum for the downloaded file, only enabled when it has value
    """
    if root is None:
        root = os.path.expanduser('~/.towhee/checkpoints')
    os.makedirs(root, exist_ok=True)
    with request.urlopen(url) as r:
        d = r.headers['content-disposition']
    if d:
        filename = d.split('=')[-1].split(' ')[-1]
        if filename.startswith(('UTF-', 'utf-')):
            filename = filename[5:]
        for char in ['\'', '"', '\\', ',']:
            filename = filename.replace(char, '')
    else:
        filename = os.path.basename(url)

    if hash_prefix == 'default':
        hash_prefix = url.split('/')[-2]

    # Check target file
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f'{download_target} exists and is not a regular file.')

    if os.path.isfile(download_target):
        if hash_prefix:
            if checksum(download_target, hash_prefix):
                warnings.warn(f'{download_target} exists and passed checksum. Using the existed one.')
                return download_target
            else:
                warnings.warn(f'{download_target} exists, '
                              f'but the checksum does not match; re-downloading file.')
        else:
            warnings.warn(f'{download_target} exists and skipped checksum. Using the existed one.')
            return download_target

    # Download from url
    with request.urlopen(url) as source, open(download_target, 'wb') as output:
        with tqdm(total=int(source.info().get('Content-Length')),
                  ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    # Checksum if hash_prefix is not None
    if hash_prefix:
        assert checksum(download_target, hash_prefix), \
            'Model has been downloaded but the hash checksum does not not match.'

    return download_target


def checksum(filepath, hash_prefix):
    """
    Check hash value of a file.

    Args:
        filepath (`str`): path of local file
        hash_prefix (`str`): prefix of hash value to compare with
    """
    with open(filepath, 'rb') as f:
        len_prefix = len(hash_prefix)
        data = f.read()
        if hashlib.sha256(data).hexdigest()[:len_prefix] == hash_prefix:
            return True
        elif hashlib.md5(data).hexdigest()[:len_prefix] == hash_prefix:
            return True
        else:
            return False
