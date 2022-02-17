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

import sys
import shutil
import argparse
from pathlib import Path
from requests.exceptions import HTTPError
from urllib.parse import urlsplit

from towhee.hub.operator_manager import OperatorManager
from towhee.hub.pipeline_manager import PipelineManager
from towhee.utils.hub_utils import HubUtils
from towhee.utils.hub_file_utils import HubFileUtils
from towhee.utils.repo_normalize import RepoNormalize
from towhee.utils.git_utils import GitUtils

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('uri', help='Repo uri, such as author/repo-name or repo-name(author defaults to login account)')
parser.add_argument('-d', '--dir', default='.', help='optional, directory to the Repo file, defaults to \'.\'')
parser.add_argument('--local', action='store_true', help='optional, create and init repo in local')
parser.add_argument('--plain', action='store_true', help='optional, just create repo with init file')


class RepoCommand:
    """
    Implementation for subcmd `towhee develop` and `towhee install`.
    Setup repo to `towheeoperator.{self._args.namespace}_{self._args.repo_name}' package with pypi methods.
    """

    def __init__(self, args) -> None:
        self._args = args
        self.file = HubFileUtils()
        self.hub = HubUtils()

    def __call__(self) -> None:
        self.uri = RepoNormalize(self._args.uri).parse_uri()
        self.token = self.file.get_token()
        if not self._args.local:
            if self.token is None:
                print('You have not log in, please run `towhee login` first.')
                sys.exit()
            else:
                author = self.hub.get_user(self.token).json()['username']
                if '/' not in urlsplit(self._args.uri):
                    self._args.uri = f'{author}/{self._args.uri}'
                self.uri = RepoNormalize(self._args.uri).parse_uri()
                print(f'creating repo with username: {self.uri.author}, and repo name: {self.uri.repo}\n')
                if author != self.uri.author:
                    print(f'Authenticated user: {author} does not match the username: {self.uri.author}, please specify the correct author.')
                    sys.exit()
        try:
            if self._args.action == 'create-op':
                self.create_op()
            if self._args.action == 'create-pipeline':
                self.create_pipeline()
            elif self._args.action == 'generate-yaml':
                print('Generating yaml for repo...')
                self.generate_yaml()
                print('Done')
        except HTTPError as e:
            print(e)

    @staticmethod
    def install(subparsers):
        op_parser = argparse.ArgumentParser(add_help=False)
        op_parser.add_argument('-t', '--type', choices=['pyop', 'nnop'], default='nnop',
                               help='optional, operator repo type in [\'pyop\', \'nnop\'] for init file, defaults to \'nnop\'')
        op_parser.add_argument('-f', '--framework', default='pytorch', help='optional, framework of nnoperator, defaults to \'pytorch\'')

        subparsers.add_parser('create-op', parents=[op_parser, parser], help='hub-repo command: create operator and generate init file')

    def create_op(self):
        op_manager = OperatorManager(self.uri.author, self.uri.repo)
        repo_path = Path(self._args.dir) / self.uri.repo
        if not self._args.local:
            self.create_repo(op_manager, repo_path)
        else:
            repo_path.mkdir(parents=True)
        if self._args.local or not self._args.plain:
            self.init_repo(op_manager, repo_path)

    def create_pipeline(self):
        pipeline_manager = PipelineManager(self.uri.author, self.uri.repo)
        repo_path = Path(self._args.dir) / self.uri.repo
        if not self._args.local:
            self.create_repo(pipeline_manager, repo_path)
        else:
            repo_path.mkdir(parents=True)
        if self._args.local or not self._args.plain:
            self.init_repo(pipeline_manager, repo_path, True)

    def create_repo(self, manager, repo_path):
        link = self.uri.full_uri.split('?')[0]
        if manager.exists():
            print(f'Noting {self.uri.author}/{self.uri.repo} repo already exists: {link}\n')
        else:
            manager.create_with_token(self.token)
            print(f'Successfully create Operator in hub: {link}\n')
        if not self._args.plain:
            GitUtils(self.uri.author, self.uri.repo).clone(local_repo_path=repo_path)

    def init_repo(self, manager, repo_path, is_pipeline=False):
        print('\nInitializing the repo file structure...\n')
        if is_pipeline:
            temp_path = Path(self._args.dir) / 'pipeline_template'
            GitUtils('towhee', 'pipeline-template').clone(local_repo_path=temp_path)
            manager.init_pipeline(temp_path, repo_path)
        elif self._args.type == 'pyop':
            temp_path = Path(self._args.dir) / 'pyoperator_template'
            GitUtils('towhee', 'pyoperator-template').clone(local_repo_path=temp_path)
            manager.init_pyoperator(temp_path, repo_path)
        elif self._args.type == 'nnop':
            temp_path = Path(self._args.dir) / 'nnoperator_template'
            GitUtils('towhee', 'nnoperator-template').clone(local_repo_path=temp_path)
            manager.init_nnoperator(temp_path, repo_path, self._args.framework)
        shutil.rmtree(str(temp_path))

    def generate_yaml(self):
        repo_manager = OperatorManager(self.uri.author, self.uri.repo)
        repo_manager.generate_yaml(Path(self._args.dir))


class PipeCommand:
    """
    Implementation for subcmd `towhee develop` and `towhee install`.
    Setup repo to `towheeoperator.{self._args.namespace}_{self._args.repo_name}' package with pypi methods.
    """

    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        RepoCommand(self._args)()

    @staticmethod
    def install(subparsers):
        subparsers.add_parser('create-pipeline', parents=[parser], help='hub-repo command: create pipeline and generate init file')
