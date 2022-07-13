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
from typing import Union
from towhee.engine import register
from towhee.engine.factory import ops, pipes, pipeline, DEFAULT_PIPELINES, _PipelineBuilder as Build
from towhee.hparam import param_scope
from towhee.hparam import HyperParameter as Document
from towhee.functional import DataCollection, State, Entity, DataFrame
from towhee.connectors import Connectors as connectors

from towhee.functional import glob, glob_zip
from towhee.functional import read_csv, read_json, read_camera
from towhee.functional import read_video, read_audio
from towhee.functional import read_zip, from_df
from towhee.functional import dc, api, dummy_input

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
    'DataFrame',
    'State',
    'Entity',
    'connectors',
    #
    'glob',
    'glob_zip',
    'from_df',
    'read_audio',
    'read_camera',
    'read_csv',
    'read_json',
    'read_video',
    'read_zip',
    'dc',
    'api',
    'dummy_input'
]


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


def plot(img1: Union[str, list], img2: list = None):
    from towhee.utils.plot_utils import plot_img  # pylint: disable=C
    if not img2:
        plot_img(img1)
    else:
        assert len(img1) == len(img2)
        for i in range(len(img1)):
            plot_img(img1[i])
            plot_img(img2[i])


def build_docker_image(dc_pipeline, image_name, format_priority, cuda):
    from towhee.serve.triton.docker_image_builder import DockerImageBuilder  # pylint: disable=import-outside-toplevel
    DockerImageBuilder(dc_pipeline, image_name, format_priority, cuda).build()
