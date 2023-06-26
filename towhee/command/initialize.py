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

import shutil
import argparse
from pathlib import Path

from towhee.hub.repo_manager import REPO_TEMPLATE
from towhee.hub.operator_manager import OperatorManager
from towhee.utils.repo_normalize import RepoNormalize


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('uri', help='Repo uri, in the form of <repo-author>/<repo-name>.')
parser.add_argument('-d', '--dir', default='.', help='optional, directory to the operator, defaults to current working directory.')
parser.add_argument('-t', '--type', choices=['pyop', 'nnop'], default='pyop', help='optional, operator type, defaults to \'pyop\'.')


class InitCommand:  # pragma: no cover
    """
    Implementation for subcmd `towhee init`.
    Create an Operator in towhee hub.
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        self.uri = RepoNormalize(self._args.uri).parse_uri()
        self.init_op()

    def init_op(self):
        op_manager = OperatorManager(self.uri.author, self.uri.repo)
        repo_path = Path(self._args.dir)
        self.init_repo(op_manager, repo_path)

    def init_repo(self, manager, repo_path):
        print('\nInitializing the repo file structure...\n')
        if self._args.type == 'pyop':
            temp_path = Path(self._args.dir) / (REPO_TEMPLATE['pyoperator'] + '_tmp')
            OperatorManager('towhee', REPO_TEMPLATE['pyoperator']).download(local_repo_path=temp_path, tag='main', install_reqs=False)
            manager.init_pyoperator(temp_path, repo_path)
        elif self._args.type == 'nnop':
            temp_path = Path(self._args.dir) / (REPO_TEMPLATE['nnoperator'] + '_tmp')
            OperatorManager('towhee', REPO_TEMPLATE['nnoperator']).download(local_repo_path=temp_path, tag='main', install_reqs=False)
            manager.init_nnoperator(temp_path, repo_path)
        shutil.rmtree(str(temp_path))

    @staticmethod
    def install(subparsers):
        subparsers.add_parser('init', parents=[parser], help='Init operator and generate template file.')
