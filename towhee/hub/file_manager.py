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


from towhee.engine.singleton import singleton
from towhee.engine import DEFAULT_LOCAL_CACHE_ROOT
from towhee.hub.hub_tools import download_repo

class FileManagerConfig:
    """
    Global FileManager config

    This class dictates which cache locations to look through and in what order to check
    for files. This config is ultimately used with FileManager.

    Args:
        set_default_cache (`str` | `Path`):
            The default cache to check in, if nothing supplied, the default cache of $HOME/.towhee
            will be used.
    """

    def __init__(self, set_default_cache: Union[str, Path] = DEFAULT_LOCAL_CACHE_ROOT):
        # TODO: #1 Deal with specifying cache priority per pipeline?
        self._default_cache = Path(set_default_cache)
        self._cache_paths = [self._default_cache]
        self._cache_operators = []
        self._cache_pipelines = []

    @property
    def default_cache(self):
        return self._default_cache

    @property
    def cache_paths(self):
        return self._cache_paths

    @property
    def cache_operators(self):
        return self._cache_operators

    @property
    def cache_pipelines(self):
        return self._cache_pipelines


    def add_cache_path(self, insert_path: Union[str, Path, List[Union[str, Path]]]):
        """
        Add a cache location to the front. Most recently added paths will be
        checked first.

        Args:
            insert_path (`str` | `Path | `list[str | Path]`):
                The path that you are trying to add. Accepts multiple inputs at once.
        """
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
        for path in remove_path:
            try:
                self._cache_paths.remove(Path(path))
            except ValueError:
                # TODO: Log this instead of print.
                print(str(Path(path)) + ' not found.')

    def reset_cache_path(self, use_def_cache: bool = True):
        """
        Remove all cache locations.

        Args:
            use_def_cache (`bool`):
                Once emptied, should the default cache be readded.
        """
        if use_def_cache:
            self._cache_paths = [self._default_cache]
        else:
            self._cache_paths = []

    def cache_local_operator(self, path: Union[str, Path], overwrite: bool = True, cache: Union[str, Path] = None):
        """
        Add a local operator to a cache. In order for operators to be visible to Towhee
        they need to be cached.

        Args:
            path (`str` | `Path`):
                The path to the operator or its directory.
            overwrite (`bool`):
                Whether to overwrite already existing operator in cache.
            cache (`str` | `Path`):
                Optional location to cache the opeartor.
        """
        if cache is None:
            cache = self._default_cache
        op = {}
        op['path'] = Path(path)
        op['name'] = op['path'].name
        op['cache'] = Path(cache)
        op['overwrite'] = overwrite

        if op not in self._cache_operators:
            self._cache_operators.append(op)
        else:
            print('Already in add queue.')

    def cache_local_pipeline(self, path: Union[str, Path], overwrite: bool = True, cache: Union[str, Path] = None):
        """
        Add a local pipeline to a cache. In order for pipelines to be visible to Towhee
        they need to be cached.

        Args:
            path (`str` | `Path`):
                The path to the pipeline or its directory.
            overwrite (`bool`):
                Whether to overwrite already existing operator in cache.
            cache (`str` | `Path`):
                Optional location to cache the opeartor.
        """
        if cache is None:
            cache = self._default_cache
        op = {}
        op['path'] = Path(path)
        op['name'] = op['path'].name
        op['cache'] = Path(cache)
        op['overwrite'] = overwrite

        if op not in self._cache_pipelines:
            self._cache_pipelines.append(op)
        else:
            print('Already in add queue.')

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
        self._pipelines = []
        self._operator_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        self._cache_locals()


    # Move to a utils location?
    def _cache_name(self, name: str, author: str, branch: str):
        # If pointing directly to pipeline file.
        if name.endswith('.yaml'):
            return name[:-5] + '&' + author + '$' + branch
        # If pointing directly to operator file.
        elif name.endswith('.py'):
            return name[:-3] + '&' + author + '$' + branch
        else:
            return name + '&' + author + '$' + branch

    # TODO: filip-halt
    # Decide if to treat pipelines and operators the same. Should we allow single .py
    # for opeartor and should we allow directory for pipeline.
    def _cache_locals(self):
        self._pipeline_lock.acquire()  # pylint: disable=consider-using-with
        for pipe in self._config.cache_pipelines:
            new_dir = self._cache_name(pipe['name'], author='local', branch='main')
            file = pipe['name']
            cache_path = pipe['cache']
            old_path = pipe['path']
            new_path = cache_path / new_dir / file

            if pipe['overwrite']:
                Path.unlink(new_path, missing_ok=True)

            if new_path.is_file():
                # TODO: filip-halt
                # Error logging.
                print('Did not overwrite.')
            else:
                # TODO: filip-halt
                # Figure out exception types.
                new_path.parent.mkdir(parents=True, exist_ok=True)
                copy2(str(old_path), str(new_path))
        self._pipeline_lock.release()

        self._operator_lock.acquire()  # pylint: disable=consider-using-with
        for op in self._config.cache_operators:
            new_dir = self._cache_name(op['name'], author='local', branch='main')
            cache_path = op['cache']
            old_path = op['path']
            new_path = cache_path / new_dir

            if op['overwrite'] and new_path.is_dir():
                rmtree(new_path)

            if new_path.is_dir():
                # TODO: filip-halt
                # Error logging.
                print('Did not overwrite.')
            else:
                # TODO: Figure out exception types
                copytree(str(old_path), str(new_path))
        self._operator_lock.release()

    def get_pipeline(self, pipeline: str, branch: str = 'main', redownload: bool = False, install_reqs = True):
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
        self._pipeline_lock.acquire()  # pylint: disable=consider-using-with
        pipeline_split = pipeline.split('/')
        # For now assuming all piplines will be classifed as 'author/repo'.
        if len(pipeline_split) != 2:
            raise ValueError(
                '''Incorrect pipeline format, should be '<author>/<pipeline_repo>'.'''
            )
        author = pipeline_split[0]
        repo = pipeline_split[1]

        pipeline_path = self._cache_name(repo, author, branch) + '/' + repo + '.yaml'

        file_path = self._config.default_cache / pipeline_path

        for path in self._config.cache_paths:
            path = path / pipeline_path
            if path.is_file():
                file_path = path
                found_existing = True
                break

        if author == 'local' and found_existing is False:
            # TODO: filip-halt
            # Error logging.
            print('Local file not found, has it been imported?')
            self._pipeline_lock.release()
            return None
        elif author == 'local' and found_existing is True:
            self._pipeline_lock.release()
            return file_path

        if redownload:
            # TODO: filip-halt
            # Error logging.
            rmtree(file_path.parent)

        if file_path.is_file() is False:
            download_repo(author, repo, branch, str(file_path.parent), install_reqs=install_reqs)

        self._pipeline_lock.release()
        return file_path

    def get_operator(self, operator: str, branch: str = 'main', redownload: bool = False, install_reqs = True):
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
        self._operator_lock.acquire()  # pylint: disable=consider-using-with
        operator_split = operator.split('/')
        # For now assuming all piplines will be classifed as 'author/repo'.
        if len(operator_split) != 2:
            raise ValueError(
                '''Incorrect operator format, should be '<author>/<operator_repo>'.'''
            )
        author = operator_split[0]
        repo = operator_split[1]

        operator_path = self._cache_name(repo, author, branch) + '/' + repo + '.py'

        file_path = self._config.default_cache / operator_path

        for path in self._config.cache_paths:
            path = path / operator_path
            if path.is_file():
                file_path = path
                found_existing = True
                break

        if author == 'local' and found_existing is False:
            # TODO: filip-halt
            # Error logging.
            print('Local file not found, has it been imported?')
            self._operator_lock.release()
            return None
        elif author == 'local' and found_existing is True:
            self._operator_lock.release()
            return file_path

        if redownload:
            # TODO: filip-halt
            # Error logging.
            rmtree(file_path.parent)

        if file_path.is_file() is False:
            download_repo(author, repo, branch, str(file_path.parent), install_reqs=install_reqs)

        self._operator_lock.release()
        return file_path

