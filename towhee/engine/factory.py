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
import threading
from typing import List, Tuple

from towhee.dataframe import DataFrame
from towhee.pipeline_format import OutputFormat
from towhee.engine.pipeline import Pipeline
from towhee.engine.engine import Engine, start_engine
from towhee.engine.operator_loader import OperatorLoader
from towhee.hub.file_manager import FileManager
from towhee.hparam.hyperparameter import CallTracer, param_scope
# pylint: disable=unused-import
from towhee.hub import preclude


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


class _OperatorLazyWrapper:
    """
    operator wrapper for lazy initialization.
    """
    def __init__(self, real_name: str, index: Tuple[str], tag: str = 'main', **kws) -> None:
        self._name = real_name.replace('.', '/').replace('_', '-')
        self._index = index
        self._tag = tag
        self._kws = kws
        self._op = None
        self._lock = threading.Lock()

    def __call__(self, *arg, **kws):
        with self._lock:
            if self._op is None:
                self._create_op()

        if bool(self._index):
            # Multi inputs.
            if isinstance(self._index[0], tuple):
                args = []
                for i in self._index[0]:
                    args.append(getattr(arg[0], i))
                res = self._op(*args, **kws)
            # Single input.
            else:
                args = getattr(arg[0], self._index[0])
                res = self._op(args, **kws)

            # Multi outputs.
            if isinstance(res, tuple):
                if not isinstance(self._index[1], tuple) or len(self._index[1]) != len(res):
                    raise IndexError(f'Op has {len(res)} outputs, but {len(self._index[1])} indices are given.')
                for i in range(len(res)):
                    setattr(arg[0], self._index[1][i], res[i])
            # Single output.
            else:
                setattr(arg[0], self._index[1], res)

            return arg[0]
        else:
            res = self._op(*arg, **kws)
            return res

    def train(self, *arg, **kws):
        with self._lock:
            if self._op is None:
                self._create_op()
        return self._op.train(*arg, **kws)

    def fit(self, *arg):
        self._op.fit(*arg)

    @property
    def is_stateful(self):
        with self._lock:
            if self._op is None:
                self._create_op()
        return hasattr(self._op, 'fit')

    def set_state(self, state):
        with self._lock:
            if self._op is None:
                self._create_op()
        self._op.set_state(state)

    def set_training(self, flag):
        self._op.set_training(flag)

    @property
    def function(self):
        return self._name

    @property
    def init_args(self):
        return self._kws

    @staticmethod
    def callback(real_name: str, index: Tuple[str], *arg, **kws):
        if arg and not kws:
            raise ValueError('The init args should be passed in the form of kwargs(i.e. You should specify the keywords of your init arguments.)')
        if len(arg) == 0:
            return _OperatorLazyWrapper(real_name, index, **kws)
        else:
            return _OperatorLazyWrapper(real_name, index, arg[0], **kws)

    def _create_op(self):
        """
        Instantiate the operator.
        """
        # pylint: disable=unused-variable
        with param_scope(index=self._index) as hp:
            self._op = op(self._name, self._tag, **self._kws)

DEFAULT_PIPELINES = {
    'image-embedding': 'towhee/image-embedding-resnet50',
    'image-encoding': 'towhee/image-embedding-resnet50',  # TODO: add encoders
    'music-embedding': 'towhee/music-embedding-vggish',
    'music-encoding': 'towhee/music-embedding-clmr',  # TODO: clmr -> encoder
}

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
        format_handler = OutputFormat.get_format_handler(self._pipeline.pipeline_type)
        return format_handler(out_df)

    def __repr__(self) -> str:
        return repr(self._pipeline)

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline


def pipeline(pipeline_src: str, tag: str = 'main', install_reqs: bool = True, **kwargs):
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
    """
    Build a pipeline with template variables.

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


def _ops_call_back(real_name: str, index: Tuple[str], *arg, **kws):
    try:
        return _PipelineBuilder.callback(real_name, index, *arg, **kws)
    except: # pylint: disable=bare-except
        return _OperatorLazyWrapper.callback(real_name, index, *arg, **kws)


class OpsCallTracer(CallTracer):
    """
    Entry point for creating operator instances, for example:

    >>> op_instance = ops.my_namespace.my_repo_name(init_arg1=xxx, init_arg2=xxx)

    An instance of `my_namespace`/`my_repo_name` is created. It will automatically judge if the repo is an operator or a pipeline in hub,
    but it will take a little time, you can also specify the `rtype`('pipeline' or 'operator') manually, for example:

    >>> op_instance = ops.my_namespace.my_op_name(init_arg1=xxx, init_arg2=xxx, rtype='operator')
    >>> pipe_instance = ops.my_namespace.my_pipe_name(init_arg1=xxx, init_arg2=xxx, rtype='pipeline')


    """
    def __init__(self, callback=None, path=None, index=None):
        super().__init__(callback=callback, path=path, index=index)


ops = OpsCallTracer(_ops_call_back)
