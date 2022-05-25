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
from .data_collection import DataCollection, DataFrame
from .entity import Entity
from .option import Option, Some, Empty
from towhee.hparam import HyperParameter as State

from towhee.hparam import param_scope
from towhee.hparam import dynamic_dispatch
# pylint: disable=protected-access


# def stream(iterable):
#     return DataCollection.stream(iterable)


# def unstream(iterable):
#     return DataCollection.unstream(iterable)


read_csv = DataFrame.from_csv

read_json = DataFrame.from_json

read_camera = DataCollection.from_camera

read_video = DataCollection.read_video

from_zip = DataCollection.from_zip

to_video = DataCollection.to_video


def from_df(dataframe, as_stream=False):
    if as_stream:
        return DataFrame.stream(dataframe.iterrows()).map(lambda x: Entity(**x[1].to_dict()))
    return DataFrame(dataframe)


@dynamic_dispatch
def glob(*arg): # pragma: no cover
    """
    Return a DataCollection of paths matching a pathname pattern.
    Examples:

    1. create a simple data collection;

    >>> import towhee
    >>> towhee.glob('*.jpg', '*.png').to_list() #doctest: +SKIP
    ['a.jpg', 'b.jpg', 'x.png']

    2. create a data collection of structural data.

    >>> towhee.glob['path']('*.jpg').to_list() #doctest: +SKIP
    [<Entity dict_keys(['path'])>, <Entity dict_keys(['path'])>]
    """

    index = param_scope()._index
    if index is None:
        return DataCollection.from_glob(*arg)
    return DataFrame.from_glob(*arg).map(lambda x: Entity(**{index: x}))


@dynamic_dispatch
def glob_zip(uri, pattern): # pragma: no cover
    """
    Return a DataCollection of files matching a pathname pattern from a zip archive.
    Examples:

    1. create a simple data collection;

    >>> import towhee
    >>> towhee.glob_zip('somefile.zip', '*.jpg').to_list() #doctest: +SKIP
    ['a.jpg', 'b.jpg]

    2. create a data collection of structural data.

    >>> towhee.glob_zip['path']('somefile.zip', '*.jpg').to_list() #doctest: +SKIP
    [<Entity dict_keys(['path'])>, <Entity dict_keys(['path'])>]
    """

    index = param_scope()._index
    if index is None:
        return DataCollection.from_zip(uri, pattern)
    return DataFrame.from_zip(uri, pattern).map(lambda x: Entity(**{index: x}))


@dynamic_dispatch
def api():
    """
    Serve the DataCollection as a RESTful API

    Args:
        path (str, optional): API path. Defaults to '/'.
        app (_type_, optional): The FastAPI app the API bind to, will create one if None.

    Returns:
        _type_: the app that bind to

    Examples:

    >>> from fastapi import FastAPI
    >>> from fastapi.testclient import TestClient
    >>> app = FastAPI()

    >>> import towhee
    >>> with towhee.api() as api:
    ...     app1 = (
    ...         api.map(lambda x: x+' -> 1')
    ...            .map(lambda x: x+' => 1')
    ...            .serve('/app1', app)
    ...     )

    >>> with towhee.api['x']() as api:
    ...     app2 = (
    ...         api.runas_op['x', 'x_plus_1'](func=lambda x: x+' -> 2')
    ...            .runas_op['x_plus_1', 'y'](func=lambda x: x+' => 2')
    ...            .select['y']()
    ...            .serve('/app2', app)
    ...     )

    >>> with towhee.api() as api:
    ...     app2 = (
    ...         api.parse_json()
    ...            .runas_op['x', 'x_plus_1'](func=lambda x: x+' -> 3')
    ...            .runas_op['x_plus_1', 'y'](func=lambda x: x+' => 3')
    ...            .select['y']()
    ...            .serve('/app3', app)
    ...     )

    >>> client = TestClient(app)
    >>> client.post('/app1', '1').text
    '"1 -> 1 => 1"'
    >>> client.post('/app2', '2').text
    '{"y":"2 -> 2 => 2"}'
    >>> client.post('/app3', '{"x": "3"}').text
    '{"y":"3 -> 3 => 3"}'
    """
    return DataFrame.api(index=param_scope()._index)


@dynamic_dispatch
def dc(iterable):
    """
    Return a DataCollection.

    Examples:

    1. create a simple data collection;

    >>> import towhee
    >>> towhee.dc([0, 1, 2]).to_list()
    [0, 1, 2]

    2. create a data collection of structural data.

    >>> towhee.dc['column']([0, 1, 2]).to_list()
    [<Entity dict_keys(['column'])>, <Entity dict_keys(['column'])>, <Entity dict_keys(['column'])>]
    """

    index = param_scope()._index
    if index is None:
        return DataCollection(iterable)
    return DataFrame(iterable).map(lambda x: Entity(**{index: x}))
