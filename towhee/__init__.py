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
from typing import List, Tuple, Union

from towhee.dataframe import DataFrame
from towhee.dataframe import Variable
from towhee.engine.engine import Engine, start_engine
from towhee.engine.pipeline import Pipeline
from towhee.pipeline_format import OutputFormat
from towhee.hub.file_manager import FileManagerConfig, FileManager
from towhee.engine import register, resolve
from towhee.engine.operator_loader import OperatorLoader
from towhee.hparam import param_scope, auto_param

__all__ = ['DEFAULT_PIPELINES', 'pipeline', 'register', 'resolve', 'param_scope', 'auto_param', 'Build', 'Inject']

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
        format_handler = OutputFormat.get_format_handler(self._pipeline.pipeline_type)
        return format_handler(out_df)

    def __repr__(self) -> str:
        return repr(self._pipeline)

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline


def pipeline(pipeline_src: str, tag: str = 'main', install_reqs: bool = True):
    """
    Entry method which takes either an input task or path to an operator YAML.

    A `Pipeline` object is created (based on said task) and subsequently added to the
    existing `Engine`.

    Args:
        pipeline_src (`str`):
            pipeline name or YAML file location to use.
        tag (`str`):
            Which tag to use for operators/pipelines on hub, defaults to `main`.
        install_reqs (`bool`):
            Whether to download the python packages if a requirements.txt file is included in the repo.

    Returns
        (`typing.Any`)
            The `Pipeline` output.
    """
    start_engine()

    if os.path.isfile(pipeline_src):
        yaml_path = pipeline_src
    else:
        fm = FileManager()
        p_repo = DEFAULT_PIPELINES.get(pipeline_src, pipeline_src)
        yaml_path = fm.get_pipeline(p_repo, tag, install_reqs)

    engine = Engine()
    pipeline_ = Pipeline(str(yaml_path))
    with param_scope() as hp:
        if not hp().towhee.dry_run(False):
            engine.add_pipeline(pipeline_)

    return _PipelineWrapper(pipeline_)


def op(operator_src: str, tag: str = 'main', **kwargs):
    """
    Entry method which takes either operator tasks or paths to python files or class in notebook.
    An `Operator` object is created with the init args(kwargs).
    Args:
        operator_src (`str`):
            operator name or python file location or class in notebook.
        tag (`str`):
            Which tag to use for operators on hub, defaults to `main`.
    Returns
        (`typing.Any`)
            The `Operator` output.
    """
    if isinstance(operator_src, type):
        class_op = type('operator', (operator_src, ), kwargs)
        return class_op.__new__(class_op, **kwargs)

    loader = OperatorLoader()
    if os.path.isfile(operator_src):
        op_obj = loader.load_operator_from_path(operator_src, kwargs)
    else:
        op_obj = loader.load_operator(operator_src, kwargs, tag)

    return op_obj


class Build:
    """
    Build a pipeline with template variables.
    """
    def __init__(self, **kws) -> None:
        self._kws = kws

    def pipeline(self, *arg, **kws):
        with param_scope() as hp:
            hp().variables = self._kws
            return pipeline(*arg, **kws)


class Inject:
    """
    Build a pipeline by operator injection.
    """
    def __init__(self, **kws) -> None:
        self._injections = {}
        for k, v in kws.items():
            self._injections[k] = v

    def pipeline(self, *arg, **kws):
        with param_scope() as hp:
            hp().injections = self._injections
            return pipeline(*arg, **kws)
