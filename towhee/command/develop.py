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
import os
from pathlib import Path
import argparse
from setuptools import setup

from typing import List


class DevelopRepo:
    """
    Implementation for subcmd `towhee develop` and `towhee install`.
    Setup repo to `towheeoperator.{self._args.namespace}_{self._args.repo_name}' package with pypi methods.
    """

    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        path = Path(self._args.path).resolve()
        os.chdir(str(path))
        repo_name = path.stem.replace('-', '_')
        package_name = f'towheeoperator.{self._args.namespace}_{repo_name}'
        packages = [package_name]
        for p in path.iterdir():
            if p.is_dir() and not p.name.startswith('.') and not p.name.endswith('.egg-info') and p.name not in ['towheeoperator', 'build', 'dist']:
                packages.append(f'{package_name}.{p.name}')

        requirements = self.read_requirements()
        if self._args.action == 'install':
            sys.argv = ['setup.py', 'install']
        if self._args.action == 'develop':
            sys.argv = ['setup.py', 'develop']
            symlink_path = path / 'towheeoperator'
            symlink = symlink_path / f'{self._args.namespace}_{repo_name}'
            if not symlink_path.exists():
                symlink_path.mkdir()

            if not symlink.is_symlink():
                symlink.symlink_to(path)

        setup(name=package_name,
              packages=packages,
              package_dir={package_name: '.'},
              package_data={'': ['*.txt', '*.rst']},
              install_requires=requirements,
              zip_safe=False)

    @staticmethod
    def read_requirements() -> List[str]:
        """
        Get a list in requirements.txt.

        Return:
            (`List`):
                List in ./requirements.txt.
        """
        try:
            with open('requirements.txt', encoding='utf-8') as f:
                required = f.read().splitlines()
        except FileNotFoundError:
            required = []
        return required

    @staticmethod
    def install(subparsers):
        parser = argparse.ArgumentParser()
        parser.add_argument('-n', '--namespace', default='towhee', help='repo author/namespace')
        parser.add_argument('path', default='.', help='path to the operator repo, cwd is \'.\'')
        subparsers.add_parser('develop', parents=[parser], add_help=False, description='develop operator with setup.py develop')
        subparsers.add_parser('install', parents=[parser], add_help=False, description='install operator with setup.py install')
