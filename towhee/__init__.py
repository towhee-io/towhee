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
from towhee.engine.engine import Engine, start_engine
from towhee.engine.pipeline import Pipeline
from towhee.pipeline_format import OutputFormat
from towhee.hub.file_manager import FileManagerConfig, FileManager
from towhee.engine import register
from towhee.engine.factory import ops
from towhee.hparam import param_scope
from towhee.hparam import HyperParameter as Document
from towhee.functional import DataCollection, State, Entity

from towhee.functional import glob, glob_zip
from towhee.functional import read_csv, read_json, read_camera
from towhee.functional import from_zip, from_df
from towhee.functional import dc

__all__ = [
    'DEFAULT_PIPELINES',
    'pipeline',
    'ops',
    'register',
    'param_scope',
    'Build',
    'Inject',
    'dataset',
    'Document',
    'DataCollection',
    'State',
    'Entity',
    #
    'glob',
    'glob_zip',
    'read_csv',
    'read_json',
    'read_camera',
    'from_zip',
    'from_df',
    'dc'
]

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


def dataset(name: str, *args, **kwargs) -> 'TorchDataSet':
    """
    Get a dataset by name, and pass into the custom params.

    Args:
        name (`str`):
            Name of a dataset.
        *args (`Any`):
            Arguments of the dataset construct method.
        **kwargs (`Any`):
            Keyword arguments of the dataset construct method.

    Returns:
        (`TorchDataSet`)
            The corresponding `TorchDataSet`.

    Examples:
        >>> from towhee import dataset
        >>> type(dataset('fake', size=10))
        <class 'towhee.data.dataset.dataset.TorchDataSet'>
    """
    from torchvision import datasets  # pylint: disable=import-outside-toplevel
    from towhee.data.dataset.dataset import TorchDataSet  # pylint: disable=import-outside-toplevel
    dataset_construct_map = {
        'mnist': datasets.MNIST, 'cifar10': datasets.cifar.CIFAR10, 'fake': datasets.FakeData
        # 'imdb': IMDB  # ,()
    }
    torch_dataset = dataset_construct_map[name](*args, **kwargs)
    return TorchDataSet(torch_dataset)


class Build:
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

    >>> pipe = Build(template_variable_1='new_value').pipeline('pipeline_name')
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

    Injecting is another pipeline templating mechanism that allows the use to modify
    the pipeline directly without declaring template variables.

    Examples:
    ```yaml
    operators:
    - name: operator_1
      function: namespace/operator_1         <<-- injection point
      ...
    - name: operator_2
      function: namespace/operator_2
    ```

    You can modify the pipeline directly by:

    >>> pipe = Inject(operator_1 = dict(function='my_namespace/my_op')).pipeline('pipeline_name')

    and the value at the injection point is replace by the `Inject` API.
    """
    def __init__(self, **kws) -> None:
        self._injections = {}
        for k, v in kws.items():
            self._injections[k] = v

    def pipeline(self, *arg, **kws):
        with param_scope() as hp:
            hp().injections = self._injections
            return pipeline(*arg, **kws)


def _pipeline_callback(name, index, *arg, **kws):
    name = name.replace('.', '/').replace('_', '-')
    _ = index
    return Build(**kws).pipeline(name, *arg)


pipes = param_scope().callholder(_pipeline_callback)
"""
Entry point for creating pipeline instances, for example:

>>> pipe_instance = pipes.my_namespace.my_pipeline_name(template_variable_1=xxx, template_variable_2=xxx)

An instance of `my_namespace`/`my_pipeline_name` is created, and template variables in the pipeline,z
`template_variable_1` and  `template_variable_2` are replaced with given values.
"""


def plot(img1: Union[str, list], img2: list = None):
    from towhee.utils.plot_utils import plot_img  # pylint: disable=C
    if not img2:
        plot_img(img1)
    else:
        assert len(img1) == len(img2)
        for i in range(len(img1)):
            plot_img(img1[i])
            plot_img(img2[i])
