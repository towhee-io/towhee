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


def build_docker_image(*args, **kwargs):
    """
    Wrapper for lazy import build_docker_image
    """
    return server_builder.build_docker_image(*args, **kwargs)


def build_pipeline_model(*args, **kwargs):
    """
    Wrapper for lazy import build_pipeline_model
    """
    return server_builder.build_pipeline_model(*args, **kwargs)


def DataCollection(*args, **kwargs):  # pylint: disable=invalid-name
    """
    Wrapper for lazy import DataCollection
    """
    return datacollection.DataCollection(*args, **kwargs)


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





