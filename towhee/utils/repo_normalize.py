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

import re
from urllib.parse import urlparse, parse_qsl, urlsplit
from typing import NamedTuple


class RepoNormalize:
    """
    To normalize the repo.

    Args:
        uri (`str`):
            The uri of the repo.
    """

    def __init__(self, uri: str):
        self._uri = uri
        self._scheme = 'https'
        self._netloc = 'towhee.io'
        self._author = 'towhee'
        self._ref = 'main'

    def parse_uri(self) -> NamedTuple('ParseResult', [('full_uri', str), ('author', str), ('repo', str), ('ref', str), ('repo_type', str),
                                                      ('norm_repo', str), ('module_name', str), ('class_name', str), ('scheme', str), ('netloc', str),
                                                      ('query', dict)]):
        """
        Parse the uri.

        Returns:
            (`NamedTuple[str, str, str, str, str, str, str, str, str, str, dict]`)
                Return the `full_uri` and its components: `author`, `repo`, `ref`, `repo_type`, `norm_repo`, `module_name`,
                `class_name`, `scheme`, `netloc`, `query`.
        """
        full_uri = self.get_full_uri()
        result = urlparse(full_uri)
        author, repo = result.path.split('/')[1:]
        query = dict(parse_qsl(result.query))
        ref = query.pop('ref')
        norm_repo, module_name, class_name = self.get_name(repo)
        ParseResult = NamedTuple('ParseResult', [('full_uri', str), ('author', str), ('repo', str), ('ref', str), ('repo_type', str),
                                                 ('norm_repo', str), ('module_name', str), ('class_name', str), ('scheme', str), ('netloc', str),
                                                 ('query', dict)])
        return ParseResult(full_uri, author, repo, ref, result.fragment, norm_repo, module_name, class_name, result.scheme, result.netloc, query)

    def get_full_uri(self) -> str:
        """
        Get the full uri.

        Returns:
            (`str`)
                The full uri from self._uri.
        """
        result = urlsplit(self._uri)
        path = result.path
        query = dict(parse_qsl(result.query))
        if not result.scheme:
            result = result._replace(scheme=self._scheme)
        if not result.netloc:
            result = result._replace(netloc=self._netloc)
        if 'ref' not in query:
            result = result._replace(query=f'{result.query}&ref={self._ref}')
        if path.endswith(']'):
            path = self.mapping(path)
        if '/' not in path:
            result = result._replace(path=f'/{self._author}/{path}')
        elif len(path.split('/')) == 2:
            result = result._replace(path=f'/{path}')

        full_uri = result.geturl()
        return full_uri

    def check_uri(self) -> bool:
        """
        Check if the uri matches the format.

        Returns:
            (`bool`)
                Check if passed.

        Raises:
            (`ValueError`)
                Raise error when false.
        """
        result = urlparse(self._uri)
        path = result.path.split('/')
        if len(path) == 1:
            repo = path[0]
        elif len(path) == 2:
            repo = path[1]
        elif len(path) == 3 and path[0] == '':
            repo = path[2]
        else:
            return False
        return self.check_repo(repo)

    def url_valid(self) -> bool:
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if re.match(regex, self._uri) is not None:
            return True
        return False

    @staticmethod
    def mapping(path: str) -> str:
        """
        Mapping the path(endswith']'), like repo[framework] to repo-framework.

        Args:
            path (`str`):
                The path to the uri.

        Returns:
            (`str`)
                Check if passed.

        Raises:
            (`ValueError`)
                Raise error when the path does not match the format.
        """
        try:
            if not path.endswith(']'):
                raise ValueError
            repo, framework = path.strip(']').split('[')
            path = f'{repo}-{framework}'
            return path
        except ValueError:
            raise ValueError(f'{path} does not match the \'[/author/]repo-name[framework]\' format!') from ValueError

    @staticmethod
    def check_repo(repo: str) -> bool:
        """
        Check if the repo name matches the format.

        Args:
            repo (`str`):
                Repo name.

        Returns:
            (`bool`)
                Check if matched.

        Raises:
            (`ValueError`)
                Raise error if false.
        """
        repo_list = repo.strip(']').split('[')
        if '_' not in repo and (repo.endswith(']') and len(repo_list) == 2 or len(repo_list) == 1 and not repo.endswith(']')):
            return True
        else:
            raise ValueError(f'repo: {repo} does not match the \'repo-name[framework]\' format!')

    @staticmethod
    def get_name(repo: str) -> NamedTuple('OpResult', [('repo_name', str), ('module_name', str), ('class_name', str)]):
        """
        Get the name for repo: normalized repo name, module name, and class name.

        Args:
            repo (`str`):
                Repo name.

        Returns:
            (`NamedTuple[str, str, str]`)
                Return the name for repo: `repo_name`, `module_name`, `class_name`.
        """
        repo_name = repo.replace('_', '-')
        module_name = repo_name.replace('-', '_')
        class_name = ''.join(x.capitalize() or '_' for x in repo_name.split('-'))
        OpResult = NamedTuple('OpResult', [('repo_name', str), ('module_name', str), ('class_name', str)])
        return OpResult(repo_name, module_name, class_name)

    @staticmethod
    def get_op(repo: str) -> NamedTuple('OpResult', [('repo', str), ('py_file', str), ('yaml_file', str), ('class_name', str)]):
        """
        Get the required name for operator: normalized repo name, python file name, yaml file name and class name.

        Args:
            repo (`str`):
                Repo name.

        Returns:
            (`NamedTuple[str, str, str, str]`)
                Return the required name for operator: `repo`, `py_file`, `yaml_file`, `class_name`.
        """
        repo_name = repo.replace('_', '-')
        file_name = repo_name.replace('-', '_')
        class_name = ''.join(x.capitalize() or '_' for x in repo_name.split('-'))
        OpResult = NamedTuple('OpResult', [('repo', str), ('py_file', str), ('yaml_file', str), ('class_name', str)])
        return OpResult(repo_name, f'{file_name}.py', f'{file_name}.yaml', class_name)

    @staticmethod
    def get_pipeline(repo: str) -> NamedTuple('PipelineResult', [('repo', str), ('yaml_file', str)]):
        """
        Get the required name for pipeline: normalized repo name and yaml file name.

        Args:
            repo (`str`):
                Repo name.

        Returns:
            (`NamedTuple[str, str]`)
                Return the required name for operator: `repo`, `yaml_file`.
        """
        repo_name = repo.replace('_', '-')
        file_name = repo_name.replace('-', '_')
        PipelineResult = NamedTuple('PipelineResult', [('repo', str), ('yaml_file', str)])
        return PipelineResult(repo_name, f'{file_name}.yaml')
