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

# pylint: disable=redefined-builtin
# pylint: disable=import-outside-toplevel

import sys

from towhee.runtime import register, pipe, ops, accelerate, AutoConfig, AutoPipes
from towhee.data_loader import DataLoader
from towhee.serve.triton import triton_client
from towhee.utils.lazy_import import LazyImport

# Legacy towhee._types
from towhee import types
_types = types  # pylint: disable=protected-access
sys.modules['towhee._types'] = sys.modules['towhee.types']


datacollection = LazyImport('datacollection', globals(), 'towhee.datacollection')
server_builder = LazyImport('server_builder', globals(), 'towhee.serve.server_builder')


__all__ = [
    'dataset',
    'pipe',
    'triton_client',
    'AutoConfig',
    'build_docker_image',
    'build_pipeline_model',
    'AutoConfig',
    'AutoPipes',
    'DataLoader'
]

__import__('pkg_resources').declare_namespace(__name__)


def build_docker_image(
        dc_pipeline: 'towhee.RuntimePipeline',
        image_name: str,
        cuda_version: str,
        format_priority: list,
        parallelism: int = 8,
        inference_server: str = 'triton',
    ):
    """
    Wrapper for lazy import build_docker_image.

    Args:
        dc_pipeline ('towhee.RuntimPipeline'):
            The pipeline to build as a model in the docker image.
        image_name (`str`):
            The name of the docker image.
        cuda_version (`str`):
            Cuda version.
        format_priority (`list`):
            The priority order of the model format.
        parallelism (`int`):
            The parallel number.
        inference_server (`str`):
            The inference server.

    Examples:
        >>> import towhee
        >>> from towhee import pipe, ops

        >>> p = (
        ...     pipe.input('url')
        ...         .map('url', 'image', ops.image_decode.cv2_rgb())
        ...         .map('image', 'vec', ops.image_embedding.timm(model_name='resnet50'))
        ...         .output('vec')
        ... )

        >>> towhee.build_docker_image(
        ...     dc_pipeline=p,
        ...     image_name='clip:v1',
        ...     cuda_version='11.7',
        ...     format_priority=['onnx'],
        ...     parallelism=4,
        ...     inference_server='triton'
        ... )
    """
    return server_builder.build_docker_image(dc_pipeline, image_name, cuda_version, format_priority, parallelism, inference_server)


def build_pipeline_model(
        dc_pipeline: 'towhee.RuntimePipeline',
        model_root: str,
        format_priority: list,
        parallelism: int = 8,
        server: str = 'triton'
    ):
    """
    Wrapper for lazy import build_pipeline_model.

    Args:
        dc_pipeline ('towhee.RuntimePipeline'):
            The piepline to build as a model.
        model_root (`str`):
            The model root path.
        format_priority (`list`):
            The priority order of the model format.
        parallelism (`int`):
            The parallel number.
        server (`str`):
            The server type.

    Examples:
        >>> import towhee
        >>> from towhee import pipe, ops

        >>> p = (
        ...     pipe.input('url')
        ...         .map('url', 'image', ops.image_decode.cv2_rgb())
        ...         .map('image', 'vec', ops.image_embedding.timm(model_name='resnet50'))
        ...         .output('vec')
        ... )

        >>> towhee.build_pipeline_model(
        ...     dc_pipeline=p,
        ...     model_root='models',
        ...     format_priority=['onnx'],
        ...     parallelism=4,
        ...     server='triton'
        ... )
    """
    return server_builder.build_pipeline_model(dc_pipeline, model_root, format_priority, parallelism, server)


def DataCollection(data):  # pylint: disable=invalid-name
    """
    Wrapper for lazy import DataCollection

    DataCollection is a pythonic computation and processing framework for unstructured
    data in machine learning and data science. It allows a data scientist or researcher
    to assemble data processing pipelines and do their model work (embedding,
    transforming, or classification) with a method-chaining style API.

    Args:
        data ('towhee.runtime.DataQueue'):
            The data to be stored in DataColletion in the form of DataQueue.
    """
    return datacollection.DataCollection(data)


def dataset(name: str, *args, **kwargs) -> 'TorchDataSet':
    """Get a dataset by name, and pass into the custom params.
    Args:
        name (str): Name of a dataset.
        *args (any): Arguments of the dataset construct method.
        **kwargs (any): Keyword arguments of the dataset construct method.
    Returns:
        TorchDataSet: The corresponding `TorchDataSet`.
    Examples:
        >>> from towhee import dataset
        >>> type(dataset('fake', size=10))
        <class 'towhee.data.dataset.dataset.TorchDataSet'>
    """
    from torchvision import datasets
    from towhee.data.dataset.dataset import TorchDataSet
    dataset_construct_map = {
        'mnist': datasets.MNIST, 'cifar10': datasets.cifar.CIFAR10, 'fake': datasets.FakeData
        # 'imdb': IMDB  # ,()
    }
    torch_dataset = dataset_construct_map[name](*args, **kwargs)
    return TorchDataSet(torch_dataset)





