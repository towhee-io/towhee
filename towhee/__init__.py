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
from typing import Any, Tuple
import os

from towhee.dataframe import DataFrame
from towhee.dataframe import Variable
from towhee.engine.engine import Engine, EngineConfig, start_engine
from towhee.engine.pipeline import Pipeline


__all__ = [
    'DEFAULT_PIPELINES',
    'pipeline'
]


DEFAULT_PIPELINES = {
    'image-embedding': 'resnet50-embedding'
}

_PIPELINE_CACHE_ENV = 'PIPELINE_CACHE'


class _PipelineWrapper:
    """A wrapper class around `Pipeline` which prevents users from having to create
    `DataFrame` instances by hand.

    Args:
        pipeline: `towhee.Pipeline`
            Base `Pipeline` instance for which this object will provide a wrapper for.
    """

    def __init__(self, pipeline_: Pipeline):
        self._pipeline = pipeline_

    def __call__(self, *args) -> Tuple[Any]:
        """Wraps the input arguments around a `Dataframe` for Pipeline.__call__().
        """

        # Create `Variable` tuple from input arguments.
        vargs = []
        for arg in args:
            vtype = type(arg).__name__
            vargs.append(Variable(vtype, arg))
        vargs = tuple(vargs)

        # Process the data through the pipeline.
        in_df = DataFrame('_in_df')
        in_df.put(vargs)
        out_df = self._pipeline(in_df)

        # Extract values from output tuple
        res = []
        for v in out_df.get(0, out_df.size)[1][0]:
            res.append(v.value)

        return tuple(res)


def _get_pipeline_cache(cache_path: str):
    if not cache_path:
        cache_path = os.environ.get(_PIPELINE_CACHE_ENV) if os.environ.get(
            _PIPELINE_CACHE_ENV) else Path.home() / '.towhee/pipelines'
    return Path(cache_path)


def _get_hello_towhee_pipeline():
    return Path(__file__).parent / 'tests/test_util/resnet50_embedding.yaml'


def pipeline(task: str, cache: str = None):
    """Entry method which takes either an input task or path to an operator YAML.

    Args:
        task: (`str`)
            Task name or YAML file location to use.
    """

    # If the task name coincides with one of the default pipelines, use the YAML
    # specified by that default pipeline instead of trying to lookup a YAML in the hub
    # or cache.

    start_engine()
    # TODO (jiangjunjie) delete when hub is ready
    if task.startswith('hello_towhee'):
        yaml_path = _get_hello_towhee_pipeline()
    else:
        task = DEFAULT_PIPELINES.get(task, task)

        # Get YAML path given task name. The default cache location for pipelines is
        # $HOME/.towhee/pipelines
        # TODO(fzliu): if pipeline is not available in cache, acquire it from hub
        cache_path = _get_pipeline_cache(cache)
        yaml_path = cache_path / (task + '.yaml')

    if not yaml_path.is_file():
        raise NameError('Can not find pipeline by name %s ' % task)

    # Create `Pipeline` object given its graph representation, then add the pipeline to
    # the existing `Engine`.
    engine = Engine()
    pipeline_ = Pipeline(str(yaml_path))
    engine.add_pipeline(pipeline_)

    return _PipelineWrapper(pipeline_)
