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
from typing import Union, List
from pathlib import Path

from towhee.engine import register
from towhee.engine.factory import ops, pipeline, DEFAULT_PIPELINES
from towhee.hparam import param_scope
from towhee.hparam import HyperParameter as Document
from towhee.hub.file_manager import FileManagerConfig
from towhee.functional import DataCollection, State, Entity, DataFrame
from towhee.functional import (
    glob, glob_zip, read_csv, read_json, read_camera, read_video, read_audio, read_zip, dc, api, dummy_input, range, from_df
)
from towhee import dc2
from towhee.dc2 import pipe, build_docker_image_v2, build_pipeline_model
from towhee.serve import triton_client


# Place all functions that are meant to be called by towhee.func() here aftering importing them.
__all__ = [
    'DEFAULT_PIPELINES',
    'pipeline',
    'ops',
    'register',
    'param_scope',
    'dataset',
    'Document',
    'DataCollection',
    'DataFrame',
    'State',
    'Entity',
    'range',
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
    'dummy_input',
    'update_default_cache',
    'add_cache_path',
    'cache_paths',
    'default_cache',
    # the new interface
    'pipe',
    'dc2',
    'triton_client',
]

__import__('pkg_resources').declare_namespace(__name__)


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


def update_default_cache(default_path: Union[str, Path]):
    """Update default cache.

    Args:
        default_path (Union[str, Path]): The default cache path.

    Examples:
        >>> import towhee
        >>> towhee.update_default_cache('mock/path')
        >>> towhee.default_cache
        PosixPath('mock/path')

        >>> from towhee.engine import DEFAULT_LOCAL_CACHE_ROOT
        >>> towhee.update_default_cache(DEFAULT_LOCAL_CACHE_ROOT)
    """
    fmc = FileManagerConfig()
    fmc.update_default_cache(default_path)


def add_cache_path(insert_path: Union[str, Path, List[Union[str, Path]]]):
    """Add a cache location to the front.

    Most recently added paths will be checked first.

    Args:
        insert_path (str | Path | list[str | Path]): The path that you are trying to add. Accepts multiple inputs at once.

    Examples:
        >>> import towhee
        >>> towhee.add_cache_path('mock/path')
        >>> towhee.cache_paths[0]
        PosixPath('mock/path')
    """
    fmc = FileManagerConfig()
    fmc.add_cache_path(insert_path)


cache_paths = FileManagerConfig().cache_paths
default_cache = FileManagerConfig().default_cache


def build_docker_image(dc_pipeline, image_name, cuda, inference_server='triton'):  # pylint: disable=unused-argument
    from towhee.serve.triton.docker_image_builder import DockerImageBuilder
    DockerImageBuilder(dc_pipeline, image_name, cuda).build()
