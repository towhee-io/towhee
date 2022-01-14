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

import os, sys
import pathlib
from setuptools import setup

class DevelopCommand:
    """
    Implementation for subcmd `towhee develop`
    """
    def __init__(self, args) -> None:
        self._args = args
        print(args)
    
    def __call__(self):
        os.chdir(self._args.path)
        cwd = os.getcwd()
        operator_name = os.path.basename(cwd)
        package_name = 'towheeoperator.{}'.format(operator_name)
        print("pack op/pipeline {} in {}".format(operator_name, cwd))

        if self._args.pack:
            sys.argv = 'setup.py bdist_wheel --universal'.split()
        if self._args.install:
            sys.argv = ['setup.py', 'install']
        if self._args.develop:
            sys.argv = ['setup.py', 'develop']
        
        setup(
            name=package_name,
            packages=[package_name],
            package_dir={package_name: "."},
            package_data={"": ["*.txt", "*.rst"]},
            zip_safe=False
        )

    @staticmethod
    def install(subparsers):
        parser_develop = subparsers.add_parser('develop', help='develop op/pipeline')
        group = parser_develop.add_mutually_exclusive_group(required=True)
        group.add_argument('-d', '--develop', action='store_true', 
            help='install `egg-link` file for development')
        group.add_argument('-p', '--pack', action='store_true', 
            help='pack the op/pipeline with `wheel` format')
        group.add_argument('-i', '--install', action='store_true', 
            help='install the op/pipeline')
        parser_develop.add_argument('path', type=pathlib.Path, nargs='?', default='.',
            help='root path of op/pipeline')
