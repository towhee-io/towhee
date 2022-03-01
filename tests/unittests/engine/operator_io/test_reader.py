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

from towhee.dataframe import DataFrame
from towhee.types._frame import _Frame, FRAME
from towhee.engine.operator_io.reader import DataFrameReader, BatchFrameReader, TimeWindowReader
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
        cols = [('int', 'int'), ('str', 'str'), ('float', 'float')]
        df = DataFrame('test', cols)
        data = [(1, 'test', 0.1)]
        self.data_size = 100
        t = DfWriter(df, data_size, data=data)
        t.set_sealed_when_stop()
        t.start()
        return df

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


class TestWindowReader(unittest.TestCase):

    '''
    time window test
    '''

    def _prepare_data(self, df):
        for i in range(5000, 11000, 40):
            f = _Frame(timestamp=i)
            df.put({'test': i, FRAME:  f})

        for i in range(21000, 25000, 40):
            f = _Frame(timestamp=i)
            df.put({'test': i, FRAME: f})

        df.seal()

    def _range_equal_step(self, df):
        reader = TimeWindowReader(df, {'test': 0}, 2, 2)
        # [4, 6)
        res = reader.read()
        self.assertEqual(len(res), 25)
        self.assertEqual(res[0].test, 5000)
        self.assertEqual(res[-1].test, 5960)
        # [6, 8)
        res = reader.read()
        self.assertEqual(len(res), 50)
        self.assertEqual(res[0].test, 6000)
        self.assertEqual(res[-1].test, 7960)
        # [8, 10)
        res = reader.read()
        self.assertEqual(len(res), 50)
        self.assertEqual(res[0].test, 8000)
        self.assertEqual(res[-1].test, 9960)
        # [10, 12)
        res = reader.read()
        self.assertEqual(len(res), 25)
        self.assertEqual(res[0].test, 10000)
        self.assertEqual(res[-1].test, 10960)

        # [20, 22)
        res = reader.read()
        self.assertEqual(len(res), 25)
        self.assertEqual(res[0].test, 21000)
        self.assertEqual(res[-1].test, 21960)
        # [22, 24)
        res = reader.read()
        self.assertEqual(len(res), 50)
        self.assertEqual(res[0].test, 22000)
        self.assertEqual(res[-1].test, 23960)
        # [24, 26)
        res = reader.read()
        self.assertEqual(len(res), 25)
        self.assertEqual(res[0].test, 24000)
        self.assertEqual(res[-1].test, 24960)

        # exception
        with self.assertRaises(StopIteration):
            reader.read()

    def _range_less_than_step(self, df):
        reader = TimeWindowReader(df, {'test': 0}, 2, 3)
        # [6, 8)
        res = reader.read()
        self.assertEqual(len(res), 50)
        self.assertEqual(res[0].test, 6000)
        self.assertEqual(res[-1].test, 7960)
        # [9, 11)
        res = reader.read()
        self.assertEqual(len(res), 50)
        self.assertEqual(res[0].test, 9000)
        self.assertEqual(res[-1].test, 10960)

        # [21, 23)
        res = reader.read()
        self.assertEqual(len(res), 50)
        self.assertEqual(res[0].test, 21000)
        self.assertEqual(res[-1].test, 22960)

        # [24, 26]
        res = reader.read()
        self.assertEqual(len(res), 25)
        self.assertEqual(res[0].test, 24000)
        self.assertEqual(res[-1].test, 24960)

        # exception
        with self.assertRaises(StopIteration):
            reader.read()

    def _range_greater_than_step(self, df):
        reader = TimeWindowReader(df, {'test': 0}, 3, 2)
        # [4, 7)
        res = reader.read()
        self.assertEqual(len(res), 50)
        self.assertEqual(res[0].test, 5000)
        self.assertEqual(res[-1].test, 6960)

        # [6, 9)
        res = reader.read()
        self.assertEqual(len(res), 75)
        self.assertEqual(res[0].test, 6000)
        self.assertEqual(res[-1].test, 8960)

        # [8, 11)
        res = reader.read()
        self.assertEqual(len(res), 75)
        self.assertEqual(res[0].test, 8000)
        self.assertEqual(res[-1].test, 10960)

        # [10, 13)
        res = reader.read()
        self.assertEqual(len(res), 25)
        self.assertEqual(res[0].test, 10000)
        self.assertEqual(res[-1].test, 10960)

        # [20, 23)
        res = reader.read()
        self.assertEqual(len(res), 50)
        self.assertEqual(res[0].test, 21000)
        self.assertEqual(res[-1].test, 22960)

        # [22, 25)
        res = reader.read()
        self.assertEqual(len(res), 75)
        self.assertEqual(res[0].test, 22000)
        self.assertEqual(res[-1].test, 24960)

        # [24, 27)
        res = reader.read()
        self.assertEqual(len(res), 25)
        self.assertEqual(res[0].test, 24000)
        self.assertEqual(res[-1].test, 24960)

        # exception
        with self.assertRaises(StopIteration):
            reader.read()

        # self.assertEqual(df.physical_size, 0)

    def test_equal_slideingwindow(self):
        df = DataFrame('test', [('test', 'int')])
        self._prepare_data(df)
        self._range_equal_step(df)

    def test_less_slideingwindow(self):
        df = DataFrame('test', [('test', 'int')])
        self._prepare_data(df)
        self._range_less_than_step(df)

    def test_greater_slideingwindow(self):
        df = DataFrame('test', [('test', 'int')])
        self._prepare_data(df)
        self._range_greater_than_step(df)

    def test_multithread_equal(self):
        df = DataFrame('test', [('test', 'int')])
        write_runner = MultiThreadRunner(target=self._prepare_data, args=(df,), thread_num=1)
        read_runner = MultiThreadRunner(target=self._range_equal_step, args=(df,), thread_num=1)
        write_runner.start()
        read_runner.start()
        write_runner.join()
        read_runner.join()
        df.gc()
        self.assertEqual(len(df), 0)


    def test_multithread_less(self):
        df = DataFrame('test', [('test', 'int')])
        write_runner = MultiThreadRunner(target=self._prepare_data, args=(df,), thread_num=1)
        read_runner = MultiThreadRunner(target=self._range_less_than_step, args=(df,), thread_num=1)
        write_runner.start()
        read_runner.start()
        write_runner.join()
        read_runner.join()
        df.gc()
        self.assertEqual(len(df), 0)

    def test_multithread_greater(self):
        df = DataFrame('test', [('test', 'int')])
        write_runner = MultiThreadRunner(target=self._prepare_data, args=(df,), thread_num=1)
        read_runner = MultiThreadRunner(target=self._range_greater_than_step, args=(df,), thread_num=1)
        write_runner.start()
        read_runner.start()
        write_runner.join()
        read_runner.join()
        df.gc()
        self.assertEqual(len(df), 0)

    def test_empty(self):
        df = DataFrame('test', [('test', 'int')])
        df.seal()
        reader1 = TimeWindowReader(df, {'test': 0}, 3, 2)
        with self.assertRaises(StopIteration):
            reader1.read()

        reader2 = TimeWindowReader(df, {'test': 0}, 3, 3)
        with self.assertRaises(StopIteration):
            reader2.read()

        reader3 = TimeWindowReader(df, {'test': 0}, 3, 4)
        with self.assertRaises(StopIteration):
            reader3.read()

    def test_normal(self):
        df = DataFrame('test', [('test', 'int')])
        for i in range(0, 1000, 40):
            f = _Frame(timestamp=i)
            df.put({'test': i, FRAME: f})
        df.seal()

        reader1 = TimeWindowReader(df, {'test': 0}, 3, 2)
        res = reader1.read()
        self.assertEqual(len(res), 25)

        reader2 = TimeWindowReader(df, {'test': 0}, 3, 3)
        res = reader2.read()
        self.assertEqual(len(res), 25)

        reader3 = TimeWindowReader(df, {'test': 0}, 3, 4)
        res = reader3.read()
        self.assertEqual(len(res), 25)

if __name__ == '__main__':
    unittest.main()
