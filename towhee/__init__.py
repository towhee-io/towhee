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

from pathlib import Path
from typing import Any, Tuple, List
from shutil import rmtree
import os

from towhee.dataframe import DataFrame
from towhee.dataframe import Variable
from towhee.engine.engine import Engine, start_engine
from towhee.utils.log import engine_log
from towhee.engine.pipeline import Pipeline
from towhee.engine import LOCAL_PIPELINE_CACHE
from towhee.hub.hub_tools import download_repo
from towhee.hub.file_manager import FileManagerConfig, FileManager

__all__ = ['DEFAULT_PIPELINES', 'pipeline']

DEFAULT_PIPELINES = {
    'image-embedding': 'towhee/image_embedding_resnet50',
}

_PIPELINE_CACHE_ENV = 'PIPELINE_CACHE'


class _PipelineWrapper:
    """
    A wrapper class around `Pipeline`.

    The class prevents users from having to create `DataFrame` instances by hand.

    Args:
        pipeline (`towhee.Pipeline`):
            Base `Pipeline` instance for which this object will provide a wrapper for.
    """

    def __init__(self, pipeline_: Pipeline):
        self._pipeline = pipeline_

    def __call__(self, *args) -> List[Tuple]:
        """
        Wraps the input arguments around a `Dataframe` for Pipeline.__call__(). For
        example:
        ```
        >>> p = pipeline('some-pipeline')
        >>> result = p(arg0, arg1)
        ```
        """
        if not args:
            return []

        # Support both single-input and multi-input (via list).
        if len(args) == 1 and isinstance(args[0], list):
            inputs = args[0]
        else:
            inputs = [args]

        in_df = DataFrame('_in_df')
        for tup in inputs:
            if not isinstance(tup, tuple):
                tup = (tup, )
            row = tuple((Variable(type(e).__name__, e) for e in tup))
            in_df.put(row)
        print(in_df)
        in_df.seal()

        out_df = self._pipeline(in_df)

        # 1-tuple outputs are automatically extracted.
        res = []
        for data in out_df.get(0, out_df.size):
            if len(data) == 1:
                res.append(data[0].value)
            else:
                res.append(tuple((v.value for v in data)))

        return res


# def _get_hello_towhee_pipeline():
#     return Path(__file__).parent / 'tests/test_util/resnet50_embedding.yaml'

def _download_pipeline(cache_path: str, task: str, branch: str = 'main', force_download: bool = False):
    """
    Do the check and download logic for pipelines.

    In Towhee all the pipelines' name should follow the format of 'author/pipeline'.
    """
    task_split = task.split('/')

    # For now assuming all piplines will be classifed as 'author/repo'.
    if len(task_split) != 2:
        raise ValueError(
            '''Incorrect pipeline name format, should be '<author>/<pipeline_repo>', if local file please place into 'local/<pipeline_dir> '''
        )

    author = task_split[0]
    repo = task_split[1]
    author_path = cache_path / author
    repo_path = author_path / repo
    yaml_path = repo_path / (repo + '.yaml')

    # Avoid downloading logic if its a fully local repo.
    if author == 'local':
        return yaml_path

    download = False
    if repo_path.is_dir():
        if force_download or not yaml_path.is_file():
            rmtree(repo_path)
            download = True
    else:
        download = True

    if download:
        engine_log.info('Downloading Pipeline: %s', repo)
        download_repo(author, repo, branch, str(repo_path))

    return yaml_path


def pipeline(task: str, fmc: FileManagerConfig = FileManagerConfig(), branch: str = 'main', force_download: bool = False):
    """
    Entry method which takes either an input task or path to an operator YAML.

    A `Pipeline` object is created (based on said task) and subsequently added to the
    existing `Engine`.

    Args:
        task (`str`):
            Task name or YAML file location to use.
        fmc (`FileManagerConfig`):
            Optional file manager config for the local instance, defaults to local cache.
        branch (`str`):
            Which branch to use for operators/pipelines on hub, defaults to `main`.
        force_download (`bool`):
            Whether to redownload pipeline and operators.

    Returns
        (`typing.Any`)
            The `Pipeline` output.
    """
    fm = FileManager(fmc)

    start_engine()
    task = DEFAULT_PIPELINES.get(task, task)
    yaml_path = fm.get_pipeline(task, branch, force_download)

    engine = Engine()
    pipeline_ = Pipeline(str(yaml_path))
    engine.add_pipeline(pipeline_)

    return _PipelineWrapper(pipeline_)
