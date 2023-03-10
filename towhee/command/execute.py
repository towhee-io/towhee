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

import os
import sys
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import List
from setuptools import setup


class PackageCommand: # pragma: no cover
    """
    Implementation for subcmd `towhee package`
    """
    def __init__(self, args) -> None:
        self._args = args
        self._path = Path.home() / '.towhee' / 'towheepackageoperator'

    def __call__(self) -> None:
        path = Path(self._args.path).resolve()
        repo_name = path.stem.replace('-', '_')
        package_name = f'towheeoperator_{self._args.namespace}_{repo_name}'
        packages = [package_name]
        version = str(date.today()).replace('-', '.') +'.'+ \
            str(3600*int(datetime.now().strftime('%H'))+60*int(datetime.now().strftime('%M'))+int(datetime.now().strftime('%S')))
        requirements = self.read_requirements()

        sys.argv = ['setup.py', 'sdist', 'bdist_wheel']

        setup(name=package_name,
              version=version,
              packages=packages,
              package_dir={package_name: '.'},
              package_data={'':['*.txt','*.md']},
              install_requires=requirements,
              zip_safe=False)

    @staticmethod
    def install(subparsers):
        install = subparsers.add_parser('package', help='execute command: package operator')
        install.add_argument('-n', '--namespace', required=True, help='optional, repo author or namespace, defaults to \'towhee\'')
        install.add_argument('-p', '--path', required=False, default='.', help='optional, path to the operator repo, defaults to cwd which is \'.\'')

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

class UploadCommand: # pragma: no cover
    """
    Implementation for subcmd `towhee upload`
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        if os.path.exists('./dist'):
            if not self._args.repository:
                if not self._args.repositoryurl:
                    subprocess.check_call([sys.executable, '-m', 'twine', 'upload', 'dist/*'])
                else:
                    subprocess.check_call([sys.executable, '-m', 'twine', 'upload', '--repository-url', self._args.repositoyurl, 'dist/*'])
            else:
                subprocess.check_call([sys.executable, '-m', 'twine', 'upload', '-r', self._args.repository, 'dist/*'])
        else:
            print('dist floder not exist, please use towhee package command to package operator.')

    @staticmethod
    def install(subparsers):
        install = subparsers.add_parser('upload', help='upload command: upload operator')
        install.add_argument('-r', '--repository', required=False, \
                            help='The repository (package index) to upload the package to. Should be a section in the config file (default: pypi).'\
                                '(Can also be set via TWINE_REPOSITORY environment variable.)')
        install.add_argument('-ru', '--repositoryurl', required=False, \
                            help='The repository (package index) URL to upload the package to. This overrides --repository.' \
                                '(Can also be set via TWINE_REPOSITORY_URL environment variable.)')
