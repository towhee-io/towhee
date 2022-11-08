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

import numpy
import os
import sys
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Union, Any, List
from setuptools import setup

from towhee import pipeline
from towhee.command.s3 import S3Bucket


class ExecuteCommand: # pragma: no cover
    """
    Implementation for subcmd `towhee run`
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self):
        pipe = pipeline(self._args.pipeline)
        try:
            res = pipe(int(self._args.input))
        except ValueError:
            res = pipe(self._args.input)
        if not self._args.output:
            try:
                print(res[0][0])
            except IndexError:
                print(res)
        else:
            path = Path(self._args.output)
            self.save_result(path, res)

    @staticmethod
    def install(subparsers):
        parser_execute = subparsers.add_parser('run', help='execute command: run towhee pipeline')

        parser_execute.add_argument('-i', '--input', required=True, help='input the parameter for pipeline defaults to None')
        parser_execute.add_argument('-o', '--output', default=None, help='optional, path to the file that will be used to write results], '
                                                                         'defaults to None which will print the result')
        parser_execute.add_argument('pipeline', type=str, help='pipeline repo or path to yaml')

    @staticmethod
    def save_result(output: Union[str, Path], res: Any) -> None:
        """
        Save the results to local `output` file.

        Args:
            output (`str` | `Path`):
                The path that you are trying to save.
            res (`Any`):
                The result with any format.
        """
        file_name = Path(output) / 'towhee_output.txt'
        print(f'writing result to Path({str(output)})/towhee_output.txt')
        with open(str(file_name), 'w', encoding='utf-8') as f:
            if isinstance(res, list):
                f.write('[')
                for item in res:
                    f.write(f'{item}')
                f.write(']\n')
            elif isinstance(res, numpy.ndarray):
                numpy.savetxt(str(file_name), res)
            else:
                f.write(f'{str(res)}\n')

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

class UploadS3Command: # pragma: no cover
    """
    Implementation for subcmd `towhee uploadS3`
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        s3 = S3Bucket()
        if not s3.upload_files(self._args.pathbucket, self._args.pathlocal):
            print('upload file to s3 error, please check command.')
    @staticmethod
    def install(subparsers):
        install = subparsers.add_parser('uploadS3', help='uploadS3 command: upload model to S3')
        install.add_argument('-pb', '--pathbucket', required=True, help='bucket path to save file')
        install.add_argument('-pl', '--pathlocal', required=True, help='local path to upload file')

class LsS3Command: # pragma: no cover
    """
    Implementation for subcmd `towhee lsS3`
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        s3 = S3Bucket()
        files = s3.get_list_s3(self._args.path)
        print(files)
    @staticmethod
    def install(subparsers):
        install = subparsers.add_parser('lsS3', help='lsS3 command: show files in S3 path')
        install.add_argument('-p', '--path', required=False, help='bucket path to show files')

class DownloadS3Command:
    """
    Implementation for subcmd `towhee DownloadS3`
    """
    def __init__(self, args) -> None:
        self._args = args

    def __call__(self) -> None:
        s3 = S3Bucket()
        if not s3.download_files(self._args.pathbucket, self._args.pathlocal):
            print('download files from s3 error, please check command.')
    @staticmethod
    def install(subparsers):
        install = subparsers.add_parser('downloadS3', help='downloadS3 command: download files in S3 path')
        install.add_argument('-pb', '--pathbucket', required=True, help='bucket path to download files')
        install.add_argument('-pl', '--pathlocal', required=True, help='local path to download files')
