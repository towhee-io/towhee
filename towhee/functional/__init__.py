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
from .data_collection import DataCollection
from .entity import Entity
from .option import Option, Some, Empty
from towhee.hparam import HyperParameter as State

from towhee.hparam.hyperparameter import CallTracer


class GlobImpl(CallTracer):
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

    def __init__(self, callback=None, path=None, index=None):
        super().__init__(callback=callback, path=path, index=index)


class GlobZipImpl(CallTracer):
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

    def __init__(self, callback=None, path=None, index=None):
        super().__init__(callback=callback, path=path, index=index)


class DCImpl(CallTracer):
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

    def __init__(self, callback=None, path=None, index=None):
        super().__init__(callback=callback, path=path, index=index)

class ApiImpl(CallTracer):
    """
    Return a towhee API.

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

    def __init__(self, callback=None, path=None, index=None):
        super().__init__(callback=callback, path=None, index=index)


def stream(iterable):
    return DataCollection.stream(iterable)


def unstream(iterable):
    return DataCollection.unstream(iterable)


read_csv = DataCollection.from_csv

read_json = DataCollection.from_json

read_camera = DataCollection.from_camera

from_zip = DataCollection.from_zip


def from_df(df, as_stream=False):
    if as_stream:
        return DataCollection.stream(
            df.iterrows()).map(lambda x: Entity(**x[1].to_dict()))
    return DataCollection(df)


def _glob_call_back(_, index, *arg, **kws):
    if index is not None:
        return DataCollection.from_glob(
            *arg, **kws).map(lambda x: Entity(**{index: x}))
    else:
        return DataCollection.from_glob(*arg, **kws)


def _glob_zip_call_back(_, index, *arg, **kws):
    if index is not None:
        return from_zip(*arg, **kws).map(lambda x: Entity(**{index: x}))
    else:
        return from_zip(*arg, **kws)


def _dc_call_back(_, index, *arg, **kws):
    if index is not None:
        return DataCollection(*arg, **kws).map(lambda x: Entity(**{index: x}))
    else:
        return DataCollection(*arg, **kws)


glob = GlobImpl(_glob_call_back)

glob_zip = GlobZipImpl(_glob_zip_call_back)

dc = DCImpl(_dc_call_back)

def _api_call_back(_, index, *arg, **kws):
    #pylint: disable=unused-argument
    kws['index'] = index
    return DataCollection.api( **kws)

api = ApiImpl(_api_call_back)
