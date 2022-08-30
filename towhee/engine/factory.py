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

# pylint: disable=unused-import
# pylint: disable=dangerous-default-value

import os
import threading
from typing import Any, Dict, List, Tuple

from towhee.dataframe import DataFrame
from towhee.pipeline_format import OutputFormat
from towhee.engine.pipeline import Pipeline
from towhee.engine.engine import Engine, start_engine
from towhee.engine.operator_loader import OperatorLoader
from towhee.hub.file_manager import FileManager
from towhee.hparam.hyperparameter import dynamic_dispatch, param_scope
from towhee.hub import preclude

from .execution.base_execution import BaseExecution
from .execution.pandas_execution import PandasExecution
from .execution.stateful_execution import StatefulExecution
from .execution.vectorized_execution import VectorizedExecution


def op(operator_src: str,
       tag: str = 'main',
       arg: List[Any] = [],
       kwargs: Dict[str, Any] = {}):
    """Create the supplied operator.

    Entry method which takes either operator tasks or paths to python files or class in notebook.
    An `Operator` object is created with the init args(kwargs).

    Args:
        operator_src (str): Operator name or python file location or class in notebook.
        tag (str, optional): Which tag to use for operators on hub. Defaults to 'main'.
        arg (List[Any], optional): Operator `args` to pass in. Defaults to [].
        kwargs (Dict[str, Any], optional): Operator `kwargs` to pass in. Defaults to {}.

    Returns:
        operator: The operator.
    """
    if isinstance(operator_src, type):
        class_op = type('operator', (operator_src, ), kwargs)
        return class_op.__new__(class_op, **kwargs)

    loader = OperatorLoader()
    if os.path.isfile(operator_src):
        return loader.load_operator_from_path(operator_src, arg, kwargs)
    else:
        return loader.load_operator(operator_src, arg, kwargs, tag)


class _OperatorLazyWrapper(  #
        BaseExecution,  #
        PandasExecution,  #
        StatefulExecution,  #
        VectorizedExecution):
    """
    Operator wrapper for lazy initialization. Inherits from different execution strategies.
    """

    def __init__(self,
                 real_name: str,
                 index: Tuple[str],
                 tag: str = 'main',
                 arg: List[Any] = [],
                 kws: Dict[str, Any] = {}) -> None:
        self._name = real_name.replace('.', '/').replace('_', '-')
        self._index = index
        self._tag = tag
        self._arg = arg
        self._kws = kws
        self._op = None
        self._lock = threading.Lock()
        self._op_config = self._kws.pop('op_config', None)
        # TODO: (How to apply such config)

    def __check_init__(self):
        with self._lock:
            if self._op is None:
                #  Called with param scope in order to pass index in to op.
                with param_scope(index=self._index):
                    self._op = op(self._name,
                                  self._tag,
                                  arg=self._arg,
                                  kwargs=self._kws)
                    if hasattr(self._op, '__vcall__'):
                        self.__has_vcall__ = True

    def get_op(self):
        self.__check_init__()
        return self._op

    @property
    def op_config(self):
        self.__check_init__()
        return self._op_config

    @property
    def function(self):
        return self._name

    @property
    def init_args(self):
        return self._kws

    @staticmethod
    def callback(real_name: str, index: Tuple[str], *arg, **kws):
        return _OperatorLazyWrapper(real_name, index, arg=arg, kws=kws)


# TODO: move to different location
DEFAULT_PIPELINES = {
    'image-embedding': 'towhee/image-embedding-resnet50',
    'image-encoding': 'towhee/image-embedding-resnet50',  # TODO: add encoders
    'music-embedding': 'towhee/music-embedding-vggish',
    'music-encoding': 'towhee/music-embedding-clmr',  # TODO: clmr -> encoder
}


class _PipelineWrapper:
    """A wrapper class around `Pipeline`.

    The class prevents users from having to create `DataFrame` instances by hand.

    Args:
        pipeline (towhee.Pipeline): Base `Pipeline` instance for which this object will provide a wrapper for.
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

        cols = []
        vargs = []
        for i, arg in enumerate(args):
            vtype = type(arg).__name__
            cols.append(('Col_' + str(i), str(vtype)))
            vargs.append(arg)
        vargs = tuple(vargs)

        # Process the data through the pipeline.
        in_df = DataFrame('_in_df', cols)
        in_df.put(vargs)
        out_df = self._pipeline(in_df)
        format_handler = OutputFormat.get_format_handler(
            self._pipeline.pipeline_type)
        return format_handler(out_df)

    def __repr__(self) -> str:
        return repr(self._pipeline)

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline


def pipeline(pipeline_src: str,
             tag: str = 'main',
             install_reqs: bool = True,
             **kwargs):
    """Entry method which takes either an input task or path to an operator YAML.

    A `Pipeline` object is created (based on said task) and subsequently added to the
    existing `Engine`.

    Args:
        pipeline_src (str): Pipeline name or YAML file location to use.
        tag (str, optional):  Which tag to use for operators/pipelines on hub. Defaults to 'main'.
        install_reqs (bool, optional): Whether to download the python packages if a requirements.txt file is included in the repo.. Defaults to True.

    Returns:
        _PipelineWrapper:  The `Pipeline` output.
    """
    from_ops = kwargs['from_ops'] if 'from_ops' in kwargs else False
    start_engine()

    if os.path.isfile(pipeline_src):
        yaml_path = pipeline_src
    else:
        fm = FileManager()
        p_repo = DEFAULT_PIPELINES.get(pipeline_src, pipeline_src)
        yaml_path = fm.get_pipeline(p_repo, tag, install_reqs, from_ops)

    engine = Engine()
    pipeline_ = Pipeline(str(yaml_path))
    with param_scope() as hp:
        if not hp().towhee.dry_run(False):
            engine.add_pipeline(pipeline_)

    return _PipelineWrapper(pipeline_)


class _PipelineBuilder:
    """Build a pipeline with template variables.

    A pipeline template is a yaml file contains `template variables`,
    which will be replaced by `variable values` when createing pipeline instance.

    Examples:
    ```yaml
    name: template_name
    variables:                           <<-- define variables and default values
        template_variable_1: default_value_1
        template_variable_2: default_value_2
    ....

    operator:
        function: {template_variable_1}  <<-- refer to the variable by name
    ```

    You can specialize template variable values with the following code:

    >>> pipe = _PipelineBuilder(template_variable_1='new_value').pipeline('pipeline_name')
    """

    def __init__(self, **kws) -> None:
        self._kws = kws

    def pipeline(self, *arg, **kws):
        with param_scope() as hp:
            hp().variables = self._kws
            return pipeline(*arg, **kws)

    @staticmethod
    def callback(name, index, *arg, **kws):
        name = name.replace('.', '/').replace('_', '-')
        _ = index
        return _PipelineBuilder(**kws).pipeline(name, *arg, from_ops=True)


@dynamic_dispatch
def ops(*arg, **kws):
    """Create operator instance.

    Entry point for creating operator instances, for example:

    >>> op_instance = ops.my_namespace.my_repo_name(init_arg1=xxx, init_arg2=xxx)
    """

    # pylint: disable=protected-access
    with param_scope() as hp:
        real_name = hp._name
        index = hp._index
    return _OperatorLazyWrapper.callback(real_name, index, *arg, **kws)


@dynamic_dispatch
def pipes(*arg, **kws):
    """Create pipeline instance.

    Entry point for creating pipeline instances, for example:

    >>> pipe_instance = pipes.my_namespace.my_repo_name(init_arg1=xxx, init_arg2=xxx)
    """
    # pylint: disable=protected-access
    with param_scope() as hp:
        real_name = hp._name
        index = hp._index
    return _PipelineBuilder.callback(real_name, index, *arg, **kws)


def create_op(func,
              name: str = 'tmp',
              index: Tuple[str] = None,
              arg: List[Any] = [],
              kws: Dict[str, Any] = {}) -> None:
    # pylint: disable=protected-access
    operator = _OperatorLazyWrapper(name, index, arg=arg, kws=kws)
    operator._op = func
    return operator
