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

from towhee.functional import DataCollection, Entity
from towhee.hparam.hyperparameter import CallTracer


class GlobImpl(CallTracer):
    """
    Return a DataCollection of paths matching a pathname pattern.
    Examples:

    1. create a simple data collection;

    >>> import towhee
    >>> towhee.glob('*.jpg').to_list()
    ['a.jpg', 'b.jpg]

    2. create a data collection of structural data.

    >>> towhee.glob['path']('*.jpg').to_list()
    [{'path': 'a.jpg'}, {'path': 'b.jpg'}]
    """

    def __init__(self, callback=None, path=None, index=None):
        super().__init__(callback=callback, path=path, index=index)


class GlobZipImpl(CallTracer):
    """
    Return a DataCollection of files matching a pathname pattern from a zip archive.
    Examples:

    1. create a simple data collection;

    >>> import towhee
    >>> towhee.glob_zip('somefile.zip', '*.jpg').to_list()
    ['a.jpg', 'b.jpg]

    2. create a data collection of structural data.

    >>> towhee.glob_zip['path']('somefile.zip', '*.jpg').to_list()
    [{'path': 'a.jpg'}, {'path': 'b.jpg'}]
    """

    def __init__(self, callback=None, path=None, index=None):
        super().__init__(callback=callback, path=path, index=index)


def stream(iterable):
    return DataCollection.stream(iterable)


def unstream(iterable):
    return DataCollection.unstream(iterable)


read_csv = DataCollection.from_csv

read_json = DataCollection.from_json

read_camera = DataCollection.from_camera

from_zip = DataCollection.from_zip


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


glob = GlobImpl(_glob_call_back)

glob_zip = GlobZipImpl(_glob_zip_call_back)
