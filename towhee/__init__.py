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
from typing import List, Tuple

from towhee.dataframe import DataFrame
from towhee.dataframe import Variable
from towhee.engine.engine import Engine, start_engine
from towhee.engine.pipeline import Pipeline
from towhee.hub.file_manager import FileManagerConfig, FileManager

__all__ = ['DEFAULT_PIPELINES', 'pipeline']

DEFAULT_PIPELINES = {
    'image-embedding': 'towhee/image-embedding-resnet50',
    'image-encoding': 'towhee/image-embedding-resnet50',  # TODO: add encoders
    'music-embedding': 'towhee/music-embedding-vggish',
    'music-encoding': 'towhee/music-embedding-clmr',  # TODO: clmr -> encoder
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
    start_engine()

    if os.path.isfile(task):
        yaml_path = task
    else:
        fm = FileManager(fmc)
        task = DEFAULT_PIPELINES.get(task, task)
        yaml_path = fm.get_pipeline(task, branch, force_download)

    engine = Engine()
    pipeline_ = Pipeline(str(yaml_path))
    engine.add_pipeline(pipeline_)

    return _PipelineWrapper(pipeline_)
