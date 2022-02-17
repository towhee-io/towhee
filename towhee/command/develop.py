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
import argparse
import subprocess
from pathlib import Path
from setuptools import setup

from typing import List

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-n', '--namespace', default='towhee', help='optional, repo author or namespace, defaults to \'towhee\'')
parser.add_argument('-p', '--path', required=False, default='.', help='optional, path to the operator repo, defaults to cwd which is \'.\'')


class SetupCommand:
    """
    Implementation for subcmd `towhee develop` and `towhee install`.
    Setup repo to `towheeoperator.{self._args.namespace}_{self._args.repo_name}' package with pypi methods.
    """

    def __init__(self, args) -> None:
        self._args = args
        self._path = Path.home() / '.towhee' / 'towheeoperator'

    def __call__(self) -> None:
        path = Path(self._args.path).resolve()
        repo_name = path.stem.replace('-', '_')
        package_name = f'towheeoperator.{self._args.namespace}_{repo_name}'

        packages = [package_name]
        for p in path.iterdir():
            if p.is_dir() and not p.name.startswith('.') and not p.name.endswith('.egg-info') and p.name not in ['__pycache__', 'build', 'dist']:
                packages.append(f'{package_name}.{p.name}')

        requirements = self.read_requirements()
        if self._args.action == 'install':
            if not self._args.develop:
                os.chdir(str(path))
                sys.argv = ['setup.py', 'install']
            else:
                pypi_path = self._path / repo_name
                symlink_path = pypi_path / 'towheeoperator'
                if not symlink_path.exists():
                    symlink_path.mkdir(parents=True)

                symlink = symlink_path / f'{self._args.namespace}_{repo_name}'
                symlink.unlink(missing_ok=True)
                packages = [package_name]
                symlink.symlink_to(path)
                os.chdir(str(pypi_path))
                sys.argv = ['setup.py', 'develop']

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
        install = subparsers.add_parser('install', parents=[parser], help='setup command: install operator with setup.py')
        install.add_argument('--develop', action='store_true', help='optional, install operator with setup.py develop')


class UninstallCommand:
    """
    Implementation for subcmd `towhee uninstall`.
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        path = Path(self._args.path).resolve()
        repo_name = path.stem.replace('-', '_')
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', f'towheeoperator.{self._args.namespace}_{repo_name}'])

    @staticmethod
    def install(subparsers):
        subparsers.add_parser('uninstall', parents=[parser], help='setup command: uninstall operator with pip uninstall')
