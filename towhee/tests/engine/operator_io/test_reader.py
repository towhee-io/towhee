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
from towhee.engine.operator_io.reader import DataFrameReader, BlockMapDataFrameReader, BatchFrameReader
from towhee.engine.operator_io import create_reader

from towhee.tests.test_util.dataframe_test_util import DfWriter, MultiThreadRunner


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

    def _create_test_df(self, data_size, name = 'test'):
        df = DataFrame(name)
        data = (Variable('int', 1), Variable('str', 'test'), Variable('float', 0.1))
        t = DfWriter(df, data_size, data=data)
        t.set_sealed_when_stop()
        t.start()
        return df

    def test_block_map_reader(self):

        data_size = 100
        df = self._create_test_df(data_size)
        test = {df.name: {'df': df, 'cols': [('v1', 0), ('v2', 2)]}}
        input_order = ['v1', 'v2']

        map_reader = BlockMapDataFrameReader(test, input_order)

        q = Queue()

        runner = MultiThreadRunner(
            target=read, args=(map_reader, q), thread_num=10)

        runner.start()
        runner.join()

        count = 0
        while not q.empty():
            self.assertEqual(q.get(), {'v1': 1, 'v2': 0.1})
            count += 1
        self.assertEqual(count, data_size)

    def test_block_map_reader_multi(self):

        data_size_long = 10
        df_long = self._create_test_df(data_size_long, name = 'long')
        data_size_short = 5
        df_short = self._create_test_df(data_size_short, name = 'short')

        test = {
            df_long.name: {'df': df_long, 'cols': [('v1_long', 0), ('v2_long', 2)]},
            df_short.name: {'df': df_short, 'cols': [('v1_short', 0), ('v2_short', 2)]}
        }

        input_order = ['v1_long', 'v2_long', 'v1_short', 'v2_short']

        map_reader = BlockMapDataFrameReader(test, input_order)

        q = Queue()

        runner = MultiThreadRunner(
            target=read, args=(map_reader, q), thread_num=10)

        runner.start()
        runner.join()

        count = 0
        while not q.empty():
            count += 1
            if count <= 5:
                self.assertEqual(q.get(), {'v1_long': 1, 'v2_long': 0.1, 'v1_short': 1, 'v2_short': 0.1})
            else:
                self.assertEqual(q.get(), {'v1_long': 1, 'v2_long': 0.1, 'v1_short': None, 'v2_short': None})
        self.assertEqual(count, data_size_long)

    def test_batch_reader(self):

        data_size = 100
        df = self._create_test_df(data_size)
        test = {df.name: {'df': df, 'cols': [('v1', 0), ('v2', 2)]}}
        input_order = ['v1', 'v2']
        batch_reader = BatchFrameReader(test, input_order, 3, 2)

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

    def test_batch_reader_multi(self):

        data_size_long = 100
        df_long = self._create_test_df(data_size_long, name = 'long')
        data_size_short = 50
        df_short = self._create_test_df(data_size_short, name = 'short')

        test = {
            df_long.name: {'df': df_long, 'cols': [('v1_long', 0), ('v2_long', 2)]},
            df_short.name: {'df': df_short, 'cols': [('v1_short', 0), ('v2_short', 2)]}
        }

        input_order = ['v1_long', 'v2_long', 'v1_short', 'v2_short']
        batch_reader = BatchFrameReader(test, input_order, 3, 2)

        q = Queue()

        runner = MultiThreadRunner(
            target=read, args=(batch_reader, q), thread_num=10)

        runner.start()
        runner.join()
        self.assertEqual(q.qsize(), 50)

        num = 0

        map1_1 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': 1, 'v2_short': 0.1}
        map1_2 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': 1, 'v2_short': 0.1}
        map1_3 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': 1, 'v2_short': 0.1}
        d1 = [
            namedtuple('input', map1_1.keys())(**map1_1),
            namedtuple('input', map1_2.keys())(**map1_2),
            namedtuple('input', map1_3.keys())(**map1_3)] #24

        map2_1 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': 1, 'v2_short': 0.1}
        map2_2 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': 1, 'v2_short': 0.1}
        map2_3 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': None, 'v2_short': None}
        d2 = [
            namedtuple('input', map2_1.keys())(**map2_1),
            namedtuple('input', map2_2.keys())(**map2_2),
            namedtuple('input', map2_3.keys())(**map2_3)] #1

        map3_1 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': None, 'v2_short': None}
        map3_2 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': None, 'v2_short': None}
        map3_3 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': None, 'v2_short': None}
        d3 = [
            namedtuple('input', map3_1.keys())(**map3_1),
            namedtuple('input', map3_2.keys())(**map3_2),
            namedtuple('input', map3_3.keys())(**map3_3)] # 24

        map4_1 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': None, 'v2_short': None}
        map4_2 = {'v1_long': 1, 'v2_long': 0.1, 'v1_short': None, 'v2_short': None}
        d4 = [
            namedtuple('input', map4_1.keys())(**map4_1),
            namedtuple('input', map4_2.keys())(**map4_2)] # 1

        while not q.empty():
            if num >=  49:
                self.assertEqual(q.get(), d4)
            elif num >= 25:
                self.assertEqual(q.get(), d3)
            elif num >= 24:
                self.assertEqual(q.get(), d2)
            else:
                self.assertEqual(q.get(), d1)
            num += 1

    def test_close_reader(self):
        data_size = 100
        df = self._create_test_df(data_size)
        test = {df.name: {'df': df, 'cols': [('v1', 0), ('v2', 2)]}}
        input_order = ['v1', 'v2']
        map_reader = create_reader(test, input_order, 'map')

        q = Queue()

        runner = MultiThreadRunner(
            target=read, args=(map_reader, q), thread_num=10)

        runner.start()
        map_reader.close()
        runner.join()

if __name__ == '__main__':
    unittest.main()
