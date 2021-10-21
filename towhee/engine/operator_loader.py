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


import importlib.util
from pathlib import Path
from typing import Any, Dict
from shutil import rmtree

from towhee.operator import Operator
from towhee.operator.nop import NOPOperator
from towhee.engine import LOCAL_OPERATOR_CACHE
from towhee.utils.hub_tools import download_repo


class OperatorLoader:
    """Wrapper class used to load operators from either local cache or a remote
    location.

    Args:
        cache_path: (`str`)
            Local cache path to use. If not specified, it will default to
            `$HOME/.towhee/operators`.
    """

    def __init__(self, cache_path: str = None):
        if cache_path is None:
            self._cache_path = LOCAL_OPERATOR_CACHE
        else:
            self._cache_path = Path(cache_path)

    def load_operator(self, function: str, args: Dict[str, Any]) -> Operator:
        """Attempts to load an operator from cache. If it does not exist, looks up the
        operator in a remote location and downloads it to cache instead. By standard
        convention, the operator must be called `Operator` and all associated data must
        be contained within a single directory.

        Args:
            function: (`str`)
                Origin and method/class name of the operator. Used to look up the proper
                operator in cache.
        """

        if function in ['_start_op', '_end_op']:
            return NOPOperator()

        # Lookup the path for the operator in local cache. Example directory structure:
        #  /home/user/.towhee/operators/organization-name/operator-name
        #   |_ /home/user/.towhee/operators/organization-name/operator-name/operator.py
        #   |_ /home/user/.towhee/operators/organization-name/operator-name/resnet.torch
        #   |_ /home/user/.towhee/operators/organization-name/operator-name/config.json
        # If file not there or force_download set to true, install operator

        fname, path = self._download_operator(function)

        #still checking if file exists since 'local' operators arent checked for
        if path.is_file():

            # Specify the module name and absolute path of the file that needs to be
            # imported.
            modname = 'towhee.operator.' + fname
            spec = importlib.util.spec_from_file_location(
                modname, path.resolve())

            # Create the module and then execute the module in its own namespace.
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Instantiate the operator object and return it to the caller for
            # `load_operator`. By convention, the operator class is simply the CamelCase
            # version of the snake_case operator.
            op_cls = ''.join(x.capitalize() or '_' for x in fname.split('_'))
            return getattr(module, op_cls)(**args)

        else:
            raise FileNotFoundError('Operator definition not found')

    # Figure out where to put branch info, needed for loading diff versions.
    # Currently not thread safe when downloading same repo, will most likely result in race
    # There are issues of where to to assign force_download since it may be called multiple times
    def _download_operator(self, task, branch: str = 'main', force_download: bool = False, install_reqs: bool = True):
        """Checks cache and downloads operator if necessary.
        """
        task_split = task.split('/')

        # For now assuming all operators will be classifed as 'author/repo'
        if len(task_split) != 2:
            raise ValueError(
                    '''Incorrect operator name format, should be '<author>/<operator_repo>' or 'local/<operator_repo>' ''')

        author = task_split[0]
        repo = task_split[1]
        author_path = self._cache_path / author
        repo_path = author_path / repo
        file_path = repo_path / (repo + '.py')


        # for now assuming local repos will be stored in local, change once figure out where to store it
        if author == 'local':
            return repo, file_path

        download = False

        # Check if .py exists or if we need to redownload
        if repo_path.is_dir():
            if force_download or not file_path.is_file():
                rmtree(repo_path)
                download = True
        else:
            download = True

        if download:
            print('Downloading Operator: ' + repo)
            download_repo(author, repo, branch, str(repo_path), install_reqs=install_reqs)

        return repo, file_path
