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
import os
from pathlib import Path
from typing import Union, List
from shutil import copy2, copytree, rmtree
try:
    import importlib.resources as importlib_resources
except ModuleNotFoundError:
    import importlib_resources

from towhee.utils.singleton import singleton
from towhee.engine import DEFAULT_LOCAL_CACHE_ROOT
from towhee.utils.log import engine_log
from towhee.hub.pipeline_manager import PipelineManager
from towhee.hub.operator_manager import OperatorManager
from towhee.utils.git_utils import GitUtils


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
    def _cache_name(self, repo: str, author: str, tag: str):
        try:
            file_name, file_type = repo.split('.')
        except ValueError:
            return Path(author) / repo.replace('-', '_') / tag
        # If pointing directly to pipeline or operator file.
        if file_type in ['py', 'yaml']:
            return Path(author) / file_name.replace('-', '_') / tag
        # If not pointing to a pipeline or an operator file.
        else:
            raise ValueError('Unsupported file type.')

    # TODO: filip-halt
    # Decide if to treat pipelines and operators the same. Should we allow single .py
    # for opeartor and should we allow directory for pipeline.
    def _cache_locals(self):
        with self._pipeline_lock and self._config.pipeline_lock:
            for pipe in self._config.cache_pipelines:
                new_dir = self._cache_name(pipe['name'], author='local', tag='main')
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
                new_dir = self._cache_name(op['name'], author='local', tag='main')
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

    def get_builtin_pipeline(self, file_name: str) -> str:
        with importlib_resources.path('towhee.hub.builtin.pipelines', '{}.yaml'.format(file_name)) as path:
            return path

    def get_pipeline(self, pipeline: str, tag: str, install_reqs: bool = True):
        """
        Obtain the path to the requested pipeline.

        This function will obtain the first reference to the pipeline from the cache locations.
        If no pipeline is found, this function will download it to the default cache location
        from the Towhee hub.

        Args:
            pipeline (`str`):
                The pipeline in 'author/repo' format. Author will be 'local' if locally imported.
            tag (`str`):
                Which tag to use of the pipeline. Will use 'main' if locally imported.
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

            if author == 'builtin':
                return self.get_builtin_pipeline(file_name)

            # This path leads to 'author/repo/tag/file_name.yaml'
            pipeline_path = self._cache_name(repo, author, tag) / (file_name + '.yaml')

            found_existing = False
            for path_iter in self._config.cache_paths:
                # This path leads to 'cache/pipelines/author/repo/tag/file_name.yaml'
                path = path_iter / 'pipelines' / pipeline_path
                if path.is_file():
                    file_path = path
                    found_existing = True
                    break

            if author == 'local':
                if found_existing is False:
                    msg = f'Pipeline {pipeline} not find, has it been imported?'
                    raise FileNotFoundError(msg)
                return file_path

            if not found_existing:
                file_path = self._config.default_cache / 'pipelines' / pipeline_path
                repo_path = file_path.parent

                # If yaml file missing but repo exists.
                if repo_path.is_dir():
                    engine_log.warning('Your local opreator repo %s exists, but not complete, \'.py\' file missing, please check on hub.', file_name)
                    rmtree(repo_path)
                    raise FileNotFoundError('Repo exists but \'.yaml\' file missing.')

                # If user has git, clone the repo, otherwise try to download.
                try:
                    git = GitUtils(author, repo)
                    git.clone(tag=tag, install_reqs=install_reqs, local_repo_path=repo_path)
                # If user does not have git in the system, subprocess throws FileNotFound error.
                except FileNotFoundError:
                    engine_log.warning(
                        '\'git\' not found, execute download instead of clone. ' \
                        'If you want to check updates every time you run the pipeline, please install \'git\' and remove current local cache.'
                    )
                    pipeline_manager = PipelineManager(author=author, repo=repo)
                    pipeline_manager.download(local_repo_path=file_path.parent, tag=tag, install_reqs=install_reqs)
                    return file_path

            # Check updates for repo.
            elif tag == 'main':
                repo_path = file_path.parent
                cwd = Path.cwd()
                os.chdir(repo_path)
                try:
                    git = GitUtils(author, repo)
                    if 'Your branch is behind' in git.status():
                        engine_log.warning('Your local pipeline %s is not up to date, updating to latest version...', file_name)
                        git.pull()
                except FileNotFoundError:
                    engine_log.warning(
                        '\'git\' not found, cannot update. ' \
                        'If you want to check updates every time you run the pipeline, please install \'git\' and remove current local cache.'
                    )
                os.chdir(cwd)

        return file_path

    def get_operator(self, operator: str, tag, install_reqs: bool = True):
        """
        Obtain the path to the requested operator.

        This function will obtain the first reference to the operator from the cache locations.
        If no opeartor is found, this function will download it to the default cache location
        from the Towhee hub.

        Args:
            operator (`str`):
                The operator in 'author/repo' format. Author will be 'local' if locally imported.
            tag (`str`):
                Which tag version to use of the opeartor. Will use 'main' if locally imported.
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
            # This path leads to 'author/repo/tag/file_name.yaml'
            operator_path = self._cache_name(repo, author, tag) / (file_name + '.py')

            found_existing = False
            for path_iter in self._config.cache_paths:
                # This path leads to 'cache/operators/author/repo/tag/file_name.py'
                path = path_iter / 'operators' / operator_path
                if path.is_file():
                    file_path = path
                    found_existing = True
                    break

            if author == 'local':
                if found_existing is False:
                    msg = f'Operator {operator} not found, has it been imported?'
                    raise FileNotFoundError(msg)
                return file_path

            if not found_existing:
                file_path = self._config.default_cache / 'operators' / operator_path
                repo_path = file_path.parent
                # If py file missing but repo exists.
                if repo_path.is_dir():
                    engine_log.error('Your local opreator repo %s exists, but not complete, \'.py\' file missing, please check on hub.', file_name)
                    rmtree(repo_path)
                    raise FileNotFoundError('Repo exists but \'.py\' file missing.')

                # If user has git, clone the repo, otherwise try to download.
                try:
                    git = GitUtils(author, repo)
                    git.clone(tag=tag, install_reqs=install_reqs, local_repo_path=repo_path)
                # If user does not have git in the system, subprocess throws FileNotFound error.
                except FileNotFoundError:
                    engine_log.warning(
                        '\'git\' not found, execute download instead of clone. ' \
                        'If you want to check updates every time you run the pipeline, please install \'git\' and remove current local cache.'
                    )
                    operator_manager = OperatorManager(author=author, repo=repo)
                    operator_manager.download(local_repo_path=file_path.parent, tag=tag, install_reqs=install_reqs)
                    return file_path

            # Check updates fro repo.
            elif tag == 'main':
                repo_path = file_path.parent
                cwd = Path.cwd()
                os.chdir(repo_path)
                try:
                    git = GitUtils(author, repo)
                    if 'Your branch is behind' in git.status():
                        engine_log.warning('Your local operator %s is not up to date, updating to latest version...', file_name)
                        git.pull()
                except FileNotFoundError:
                    engine_log.warning(
                        '\'git\' not found, cannot update. ' \
                        'If you want to check updates every time you run the pipeline, please install \'git\' and remove current local cache.'
                    )
                os.chdir(cwd)

        return file_path
