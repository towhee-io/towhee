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
from towhee.utils.hub_tools import download_repo

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
            raise RuntimeError('Input data is empty')

        vargs = []
        for arg in args:
            vtype = type(arg).__name__
            vargs.append(Variable(vtype, arg))
        vargs = tuple(vargs)

        # Process the data through the pipeline.
        in_df = DataFrame('_in_df')
        in_df.put(vargs)
        out_df = self._pipeline(in_df)

        res = []
        it = out_df.map_iter()
        for data in it:
            # data is Tuple[Variable]
            data_value = []
            for item in data:
                data_value.append(item.value)
            res.append(tuple(data_value))
        return res


def _get_pipeline_cache(cache_path: str):
    if not cache_path:
        cache_path = os.environ.get(_PIPELINE_CACHE_ENV) if os.environ.get(_PIPELINE_CACHE_ENV) else LOCAL_PIPELINE_CACHE
    return Path(cache_path)


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
        engine_log.info('Downloading Pipeline: ' + repo)
        download_repo(author, repo, branch, str(repo_path))

    return yaml_path


def pipeline(task: str, cache: str = None, force_download: bool = False):
    """
    Entry method which takes either an input task or path to an operator YAML.

    A `Pipeline` object is created (based on said task) and subsequently added to the
    existing `Engine`.

    Args:
        task (`str`):
            Task name or YAML file location to use.
        cache (`str`):
            Cache path to use.
        force_download (`bool`):
            Whether to redownload pipeline and operators.

    Returns
        (`typing.Any`)
            The `Pipeline` output.
    """

    # If the task name coincides with one of the default pipelines, use the YAML
    # specified by that default pipeline instead of trying to lookup a YAML in the hub
    # or cache.

    start_engine()
    # TODO (jiangjunjie) delete when hub is ready
    # if task.startswith('hello_towhee'):
    #     yaml_path = _get_hello_towhee_pipeline()
    # else:
    task = DEFAULT_PIPELINES.get(task, task)

    # Get YAML path given task name. The default cache location for pipelines is
    # $HOME/.towhee/pipelines
    # TODO(fzliu): if pipeline is not available in cache, acquire it from hub
    cache_path = _get_pipeline_cache(cache)

    yaml_path = _download_pipeline(cache_path, task, force_download=force_download)

    if not yaml_path.is_file():
        raise NameError(F'Can not find pipeline by name {task}')

    engine = Engine()
    pipeline_ = Pipeline(str(yaml_path))
    engine.add_pipeline(pipeline_)

    return _PipelineWrapper(pipeline_)
