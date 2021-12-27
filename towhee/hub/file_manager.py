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

import threading
from pathlib import Path
from typing import Union, List
from shutil import copy2, copytree, rmtree

from towhee.utils.singleton import singleton
from towhee.engine import DEFAULT_LOCAL_CACHE_ROOT
from towhee.utils.log import engine_log
from towhee.hub.hub_tools import download_repo


@singleton
class FileManagerConfig():
    """
    Global FileManager config

    This class dictates which cache locations to look through and in what order to check
    for files. This config is ultimately used with FileManager.

    Args:
        set_default_cache (`str` | `Path`):
            The default cache to check in, if nothing supplied, the default cache of $HOME/.towhee
            will be used.
    """

    def __init__(self):
        # TODO: #1 Deal with specifying cache priority per pipeline?
        self._cache_paths = [DEFAULT_LOCAL_CACHE_ROOT]
        self._cache_operators = []
        self._cache_pipelines = []

        self._cache_paths_lock = threading.Lock()
        self._operators_lock = threading.Lock()
        self._pipelines_lock = threading.Lock()

    # Requires manual locking
    @property
    def default_cache(self):
        return self._cache_paths[-1]

    # Requires manual locking
    @property
    def cache_paths(self):
        return self._cache_paths

    # Requires manual locking
    @property
    def cache_operators(self):
        return self._cache_operators

    # Requires manual locking
    @property
    def cache_pipelines(self):
        return self._cache_pipelines

    @property
    def operator_lock(self):
        return self._operators_lock

    @property
    def pipeline_lock(self):
        return self._pipelines_lock

    @property
    def cache_paths_lock(self):
        return self._cache_paths_lock

    def update_default_cache(self, default_path: Union[str, Path]):
        """
        Change the default cache path.

        Args:
            insert_path (`str` | `Path`):
                The new default cache.
        """
        default_path = Path(default_path)
        with self._cache_paths_lock:
            self._cache_paths.append(default_path)

    def add_cache_path(self, insert_path: Union[str, Path, List[Union[str, Path]]]):
        """
        Add a cache location to the front. Most recently added paths will be
        checked first.

        Args:
            insert_path (`str` | `Path | `list[str | Path]`):
                The path that you are trying to add. Accepts multiple inputs at once.
        """
        with self._cache_paths_lock:
            if not isinstance(insert_path, list):
                insert_path = [insert_path]

            for path in insert_path:
                if str(path) not in [str(x) for x in self._cache_paths]:
                    self._cache_paths.insert(0, Path(path))

    # Pretty useless now but might be worth having in the future
    def remove_cache_path(self, remove_path: Union[str, Path, List[Union[str, Path]]]):
        """
        Remove a cache location.

        Args:
            remove_path (`str` | `Path | `list[str | Path]`):
                The path that you are trying to remove. Accepts multiple inputs at once.
        """
        if not isinstance(remove_path, list):
            remove_path = [remove_path]

        with self._cache_paths_lock:
            default_path = self._cache_paths[-1]
            for path in remove_path:
                try:
                    self._cache_paths.remove(Path(path))
                except ValueError:
                    engine_log.error('%s%s ', str(Path(path)), ' not found.')

            if len(self._cache_paths) <= 1:
                self._cache_paths.append(default_path)

    def reset_cache_path(self):
        """
        Remove all cache locations.

        """
        with self._cache_paths_lock:
            self._cache_paths = [self._cache_paths.pop()]

    def cache_local_operator(self, path: Union[str, Path, List[Union[str, Path]]], overwrite: bool = True, cache: Union[str, Path] = None):
        """
        Add a local operator to a cache. In order for operators to be visible to Towhee
        they need to be cached.

        Args:
            path (`str` | `Path | `list[str | Path]`):
                The path to the operator or its directory.
            overwrite (`bool`):
                Whether to overwrite already existing operator in cache.
            cache (`str` | `Path`):
                Optional location to cache the opeartor.
        """
        if not isinstance(path, list):
            path = [path]

        with self._operators_lock:
            if cache is None:
                with self._cache_paths_lock:
                    cache = self._cache_paths[-1]
            for paths in path:
                op = {}
                op['path'] = Path(paths)
                op['name'] = op['path'].name
                op['cache'] = Path(cache)
                op['overwrite'] = overwrite

                if op not in self._cache_operators:
                    self._cache_operators.append(op)

    def cache_local_pipeline(self, path: Union[str, Path, List[Union[str, Path]]], overwrite: bool = True, cache: Union[str, Path] = None):
        """
        Add a local pipeline to a cache. In order for pipelines to be visible to Towhee
        they need to be cached.

        Args:
            path (`str` | `Path | `list[str | Path]`):
                The path to the pipeline or its directory.
            overwrite (`bool`):
                Whether to overwrite already existing operator in cache.
            cache (`str` | `Path`):
                Optional location to cache the opeartor.
        """
        if not isinstance(path, list):
            path = [path]

        with self._pipelines_lock:
            if cache is None:
                with self._cache_paths_lock:
                    cache = self._cache_paths[-1]
            for paths in path:
                op = {}
                op['path'] = Path(paths)
                op['name'] = op['path'].name
                op['cache'] = Path(cache)
                op['overwrite'] = overwrite

                if op not in self._cache_pipelines:
                    self._cache_pipelines.append(op)

    def __str__(self):
        return str(self.__dict__)


@singleton
class FileManager():
    """
    The filemanager for the current instance of Towhee. In charge of dealing with multiple
    cache locations, downloading from hub, and importing local files.

    Args:
        fmc (`FileManagerConfig`):
            Accepts an optional FileManager config, once a FileManagerConfig is selected,
            it cannot be changed for the current runtime.
    """

    def __init__(self, fmc: FileManagerConfig = FileManagerConfig()):
        self._config = fmc
        # TODO: #1 seperate ranking for different pipelines?
        # self._pipelines = []
        self._operator_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        self._cache_locals()

    # Move to a utils location?
    def _cache_name(self, name: str, author: str, branch: str):
        try:
            file_name, file_type = name.split('.')
        except ValueError:
            return name.replace('-', '_') + '&' + author + '$' + branch

        # If pointing directly to pipeline or operator file.
        if file_type in ['py', 'yaml']:
            return file_name.replace('-', '_') + '&' + author + '$' + branch
        # If not pointing to a pipeline or operator file.
        else:
            raise ValueError('Unsupported file type.')

    # TODO: filip-halt
    # Decide if to treat pipelines and operators the same. Should we allow single .py
    # for opeartor and should we allow directory for pipeline.
    def _cache_locals(self):
        with self._pipeline_lock and self._config.pipeline_lock:
            for pipe in self._config.cache_pipelines:
                new_dir = self._cache_name(pipe['name'], author='local', branch='main')
                file = pipe['name']
                cache_path = pipe['cache']
                old_path = pipe['path']
                new_path = cache_path / 'pipelines' / new_dir / file

                if pipe['overwrite']:
                    Path.unlink(new_path, missing_ok=True)

                if new_path.is_file():
                    # TODO: filip-halt
                    # Error logging.
                    engine_log.info('Did not overwrite.')
                else:
                    # TODO: filip-halt
                    # Figure out exception types.
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    copy2(str(old_path), str(new_path))

        with self._operator_lock and self._config.operator_lock:
            for op in self._config.cache_operators:
                new_dir = self._cache_name(op['name'], author='local', branch='main')
                cache_path = op['cache']
                old_path = op['path']
                new_path = cache_path / 'operators' / new_dir

                if op['overwrite'] and new_path.is_dir():
                    rmtree(new_path)

                if new_path.is_dir():
                    # TODO: filip-halt
                    # Error logging.
                    engine_log('Did not overwrite.')
                else:
                    # TODO: Figure out exception types
                    copytree(str(old_path), str(new_path))

    def get_pipeline(self, pipeline: str, branch: str = 'main', redownload: bool = False, install_reqs=True):
        """
        Obtain the path to the requested pipeline.

        This function will obtain the first reference to the pipeline from the cache locations.
        If no pipeline is found, this function will download it to the default cache location
        from hub.towhee.io.

        Args:
            pipeline (`str`):
                The pipeline in 'author/repo' format. Author will be 'local' if locally imported.
            branch (`str`):
                Which branch version to use of the pipeline. Branch will be 'main' if locally imported.
            redownload (`bool`):
                Whether to delete and redownload the pipeline files
            install_reqs (`bool`):
                Whether to download the python packages if a requirements.txt file is included in the repo.

        Returns:
            (Path | None)
                Returns the path of the pipeline, None if a local pipeline isnt found.

        Raises:
            (`ValueError`):
                Incorrect pipeline format.
        """
        with self._pipeline_lock:
            pipeline_split = pipeline.split('/')
            # For now assuming all piplines will be classifed as 'author/repo'.
            if len(pipeline_split) != 2:
                raise ValueError('''Incorrect pipeline format, should be '<author>/<pipeline_repo>'.''')
            author = pipeline_split[0]
            repo = pipeline_split[1]
            file_name = repo.replace('-', '_')

            pipeline_path = self._cache_name(repo, author, branch) + '/' + file_name + '.yaml'

            file_path = self._config.default_cache / 'pipelines' / pipeline_path
            found_existing = False

            for path in self._config.cache_paths:
                path = path / 'pipelines' / pipeline_path
                if path.is_file():
                    file_path = path
                    found_existing = True
                    break

            if author == 'local':
                if found_existing is False:
                    # TODO: filip-halt
                    # Error logging.
                    engine_log.info('Local file not found, has it been imported?')
                    file_path = None
            else:
                if redownload:
                    # TODO: filip-halt
                    # Error logging.
                    rmtree(file_path.parent)

                if not file_path.is_file():
                    download_repo(author, repo, branch, str(file_path.parent), install_reqs=install_reqs)

        return file_path

    def get_operator(self, operator: str, branch: str = 'main', redownload: bool = False, install_reqs=True):
        """
        Obtain the path to the requested operator.

        This function will obtain the first reference to the operator from the cache locations.
        If no opeartor is found, this function will download it to the default cache location
        from hub.towhee.io.

        Args:
            operator (`str`):
                The operator in 'author/repo' format. Author will be 'local' if locally imported.
            branch (`str`):
                Which branch version to use of the opeartor. Branch will be 'main' if locally imported.
            redownload (`bool`):
                Whether to delete and redownload the operator files
            install_reqs (`bool`):
                Whether to download the python packages if a requirements.txt file is included in the repo.

        Returns:
            (Path | None)
                Returns the path of the operator, None if a local operator isnt found.

        Raises:
            (`ValueError`):
                Incorrect opeartor format.
        """
        with self._operator_lock:
            operator_split = operator.split('/')
            # For now assuming all piplines will be classifed as 'author/repo'.
            if len(operator_split) != 2:
                raise ValueError('''Incorrect operator format, should be '<author>/<operator_repo>'.''')
            author = operator_split[0]
            repo = operator_split[1]
            file_name = repo.replace('-', '_')

            operator_path = self._cache_name(repo, author, branch) + '/' + file_name + '.py'

            file_path = self._config.default_cache / 'operators' / operator_path
            found_existing = False

            for path in self._config.cache_paths:
                path = path / 'operators' / operator_path
                if path.is_file():
                    file_path = path
                    found_existing = True
                    break

            if author == 'local':
                if found_existing is False:
                    # TODO: filip-halt
                    # Error logging.
                    engine_log.info('Local file not found, has it been imported?')
                    file_path = None
            else:
                if redownload:
                    # TODO: filip-halt
                    # Error logging.
                    rmtree(file_path.parent)

                if file_path.is_file() is False:
                    download_repo(author, repo, branch, str(file_path.parent), install_reqs=install_reqs)
        return file_path
