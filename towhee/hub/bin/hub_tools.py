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

import argparse
from getpass import getpass
from pathlib import Path
from shutil import rmtree

from towhee.hub.operator_manager import OperatorManager
from towhee.hub.pipeline_manager import PipelineManager
from towhee.hub.repo_manager import RepoManager
from towhee.utils.git_utils import GitUtils


def get_repo_obj(args):
    if args.command == 'download':
        repo_obj = RepoManager(args.author, args.repo)
    elif args.command == 'clone':
        repo_obj = GitUtils(args.author, args.repo)
    elif args.command == 'generate-yaml' or args.classes in ['pyoperator', 'nnoperator', 'operator']:
        repo_obj = OperatorManager(args.author, args.repo)
    elif args.classes == 'pipeline':
        repo_obj = PipelineManager(args.author, args.repo)
    return repo_obj


def init_repo(args):
    repo_manager = get_repo_obj(args)
    repo_path = Path(args.dir) / args.repo.replace('-', '_')
    GitUtils(args.author, args.repo).clone(local_repo_path=repo_path)
    if args.classes == 'pipeline':
        temp_path = Path(args.dir) / 'pipeline_template'
        GitUtils('towhee', 'pipeline-template').clone(local_repo_path=temp_path)
        repo_manager.init_pipeline(temp_path, repo_path)
    elif args.classes == 'pyoperator':
        temp_path = Path(args.dir) / 'pyoperator_template'
        GitUtils('towhee', 'pyoperator-template').clone(local_repo_path=temp_path)
        repo_manager.init_pyoperator(temp_path, repo_path)
    elif args.classes == 'nnoperator':
        temp_path = Path(args.dir) / 'nnoperator_template'
        GitUtils('towhee', 'nnoperator-template').clone(local_repo_path=temp_path)
        repo_manager.init_nnoperator(temp_path, repo_path, args.framework)
    rmtree(temp_path)


# Get more usage in README.md
def main():
    parser = argparse.ArgumentParser(prog='towhee')
    ar_parser = argparse.ArgumentParser()
    ar_parser.add_argument('-a', '--author', required=True, help='Author of the Repo.')
    ar_parser.add_argument('-r', '--repo', required=True, help='Repo name.')
    c1_parser = argparse.ArgumentParser(add_help=False)
    c1_parser.add_argument('classes', choices=['operator', 'pipeline'], help='Repo class in [\'operator\', \'pipeline\'].')
    c2_parser = argparse.ArgumentParser(add_help=False)
    c2_parser.add_argument(
        '-c', '--classes', choices=['pyoperator', 'nnoperator', 'pipeline'], help='Repo class in [\'pyoperator\', \'nnoperator\', \'pipeline\'].'
    )
    f_parser = argparse.ArgumentParser(add_help=False)
    f_parser.add_argument('-f', '--framework', default='pytorch', help='The framework of nnoperator, defaults to \'pytorch\'.')
    t_parser = argparse.ArgumentParser(add_help=False)
    t_parser.add_argument('-t', '--tag', default='main', help='Repo tag, defaults to \'main\'.')
    d_parser = argparse.ArgumentParser(add_help=False)
    d_parser.add_argument('-d', '--dir', default='.', help='Directory to the Repo file, defaults to \'.\'.')

    subparsers = parser.add_subparsers(dest='command')
    create = subparsers.add_parser('create', parents=[c1_parser, ar_parser], add_help=False, description='Create Repo on Towhee hub.')
    create.add_argument('-p', '--password', nargs='?', required=True, help='Password of the author.')
    subparsers.add_parser(
        'init', parents=[ar_parser, c2_parser, d_parser, f_parser], add_help=False, description='Initialize the file for your Repo.'
    )
    subparsers.add_parser('generate-yaml', parents=[ar_parser, d_parser], add_help=False, description='Generate yaml file for your Operator Repo.')
    subparsers.add_parser('download', parents=[ar_parser, t_parser, d_parser], add_help=False, description='Download repo(without .git) to local.')
    subparsers.add_parser('clone', parents=[ar_parser, t_parser, d_parser], add_help=False, description='Clone repo to local.')

    args = parser.parse_args()
    repo_obj = get_repo_obj(args)

    if args.command == 'create':
        if not args.password:
            args.password = getpass('Password: ')
        print('Creating repo...')
        repo_obj.create(args.password)
        print('Done')
    if args.command == 'init':
        print('Clone the repo and initialize it with template...')
        init_repo(args)
        print('Done')
    elif args.command == 'generate-yaml':
        print('Generating yaml for repo...')
        repo_obj.generate_yaml(Path(args.dir))
        print('Done')
    elif args.command == 'download':
        print('Downloading repo...')
        repo_obj.download(Path(args.dir) / args.repo, args.tag, False)
        print('Done')
    elif args.command == 'clone':
        print('Cloning repo...')
        repo_obj.clone(args.tag, False, Path(args.dir) / args.repo)
        print('Done')


if __name__ == '__main__':
    main()
