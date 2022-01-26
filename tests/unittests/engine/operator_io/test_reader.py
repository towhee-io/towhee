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


import unittest
from queue import Queue
from collections import namedtuple

from towhee.dataframe import DataFrame, Variable
from towhee.engine.operator_io.reader import DataFrameReader, BatchFrameReader
from towhee.engine.operator_io import create_reader

from tests.unittests.test_util.dataframe_test_util import DfWriter, MultiThreadRunner


def read(reader: DataFrameReader, q: Queue):
    while True:
        try:
            item = reader.read()
            if item:
                q.put(item)
                continue
        except StopIteration:
            break


class TestReader(unittest.TestCase):
    """
    Reader test
    """

    def _create_test_df(self, data_size):
        df = DataFrame('test')
        data = (Variable('int', 1), Variable(
            'str', 'test'), Variable('float', 0.1))
        self.data_size = 100
        t = DfWriter(df, data_size, data=data)
        t.set_sealed_when_stop()
        t.start()
        return df

    # def test_block_map_reader(self):

    #     data_size = 100
    #     df = self._create_test_df(data_size)

    #     map_reader = BlockMapDataFrameReader(df, {'v1': 0, 'v2': 2})

    #     q = Queue()

    #     runner = MultiThreadRunner(
    #         target=read, args=(map_reader, q), thread_num=10)

    #     runner.start()
    #     runner.join()

    #     count = 0
    #     while not q.empty():
    #         self.assertEqual(q.get(), {'v1': 1, 'v2': 0.1})
    #         count += 1
    #     self.assertEqual(count, data_size)

    def test_batch_reader(self):

        data_size = 100
        df = self._create_test_df(data_size)
        batch_reader = BatchFrameReader(df, {'v1': 0, 'v2': 2}, 3, 2)

        q = Queue()

        runner = MultiThreadRunner(
            target=read, args=(batch_reader, q), thread_num=10)

        runner.start()
        runner.join()
        self.assertEqual(q.qsize(), 50)

        num = 0
        d1_map = {'v1': 1, 'v2': 0.1}
        d1 = namedtuple('input', d1_map.keys())(**d1_map)
        while not q.empty():
            if num < 49:
                self.assertEqual(q.get(), [d1] * 3)
            else:
                self.assertEqual(q.get(), [d1] * 2)
            num += 1

    def test_close_reader(self):
        data_size = 100
        df = self._create_test_df(data_size)
        map_reader = create_reader(df, 'map', {'v1': 0, 'v2': 2})

        q = Queue()

        runner = MultiThreadRunner(
            target=read, args=(map_reader, q), thread_num=10)

        runner.start()
        map_reader.close()
        runner.join()
