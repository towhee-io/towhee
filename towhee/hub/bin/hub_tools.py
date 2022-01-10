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

from towhee.hub.operator_manager import OperatorManager
from towhee.hub.pipeline_manager import PipelineManager
from towhee.hub.repo_manager import RepoManager


def get_repo_manager(args):
    if args.command == 'download' or args.command == 'clone':
        manager = RepoManager(args.author, args.repo)
    elif args.command == 'generate-yaml' or args.type == 'operator':
        manager = OperatorManager(args.author, args.repo)
    elif args.type == 'pipeline':
        manager = PipelineManager(args.author, args.repo)
    return manager


# Get more usage in README.md
def main():
    parser = argparse.ArgumentParser(prog='towheehub')
    ar_parser = argparse.ArgumentParser()
    ar_parser.add_argument('-a', '--author', required=True, help='Author of the Repo.')
    ar_parser.add_argument('-r', '--repo', required=True, help='Repo name.')
    c_parser = argparse.ArgumentParser(add_help=False)
    c_parser.add_argument('type', choices=['operator', 'pipeline'], help='Repo type/class in [\'operator\', \'pipeline\'].')
    td_parser = argparse.ArgumentParser(add_help=False)
    td_parser.add_argument('-t', '--tag', default='main', help='Repo tag, defaults to \'main\'.')
    td_parser.add_argument('-d', '--dir', default='.', help='Directory to clone the Repo file, defaults to \'.\'.')

    subparsers = parser.add_subparsers(dest='command')
    create = subparsers.add_parser('create', parents=[c_parser, ar_parser], add_help=False, description='Create Repo on Towhee hub.')
    create.add_argument('-p', '--password', nargs='?', required=True, help='Password of the author.')
    subparsers.add_parser('download', parents=[ar_parser, td_parser], add_help=False, description='Download repo file(without .git) to local.')
    subparsers.add_parser('clone', parents=[ar_parser, td_parser], add_help=False, description='Clone repo to local.')
    subparsers.add_parser('init', parents=[ar_parser, c_parser], add_help=False, description='Initialize the file structure for your Repo.')
    subparsers.add_parser('generate-yaml', parents=[ar_parser], add_help=False, description='Generate yaml file for your Operator Repo.')

    args = parser.parse_args()
    manager = get_repo_manager(args)

    init_choice = ''
    if args.command == 'create':
        if not args.password:
            args.password = getpass('Password: ')
        manager.create(args.password)
        init_choice = input('Do you want to clone and initialize it with template? [Y|n]  ')
    if args.command == 'init' or init_choice.lower() in ['yes', 'y']:
        manager.init()
    elif args.command == 'generate-yaml':
        manager.generate_yaml()
    elif args.command == 'download':
        manager.download(args.tag, args.dir)
    elif args.command == 'clone':
        manager.clone(args.tag, args.dir)


if __name__ == '__main__':
    main()
