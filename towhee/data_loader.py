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

from typing import Callable, Iterable, Union


class DataLoader:
    """
    DataLoader

    Args:
        data_source (`Uniton[Iterable, Callable]`)
            If it is a Callable type, after the Callable object is executed, it returns an Iterable.

        parser (`Callable`)：
            Convert the data read from the data source into input for the pipeline.

        batch_size (`int`)


    Examples:
        >>> from towhee import DataLoader, pipe, ops
        >>> p = pipe.input('num').map('num', 'ret', lambda x: x + 1).output('ret')
        >>> for data in DataLoader([{'num': 1}, {'num': 2}, {'num': 3}], parser=lambda x: x['num']):
        >>>     print(p(data).to_list(kv_format=True))
        [{'ret': 2}]
        [{'ret': 3}]
        [{'ret': 4}]
    """

    def __init__(self, data_source: Union[Iterable, Callable], parser: Callable = None, batch_size: int = None):
        self._ds = data_source
        self._parser = parser if parser is not None else lambda x: x
        self._batch_size = batch_size

    def _batcher(self, ds):
        batch = []
        for data in ds:
            new_data = self._parser(data)
            batch.append(new_data)
            if len(batch) >= self._batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
            batch = []

    def _single(self, ds):
        for data in ds:
            yield self._parser(data)

    def __iter__(self):
        if callable(self._ds):
            ds = self._ds()
        elif isinstance(self._ds, Iterable):
            ds = self._ds
        else:
            raise RuntimeError("Data source only support ops or iterator")

        if self._batch_size is None:
            yield from self._single(ds)
        else:
            yield from self._batcher(ds)
