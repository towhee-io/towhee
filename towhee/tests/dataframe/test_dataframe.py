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
import time
import queue

from towhee.dataframe import DataFrame, DataFrameIterator

from towhee.tests.test_util.dataframe_test_util import DfWriter, MultiThreadRunner


class TestDataframe(unittest.TestCase):
    """
    Test dataframe basic function
    """

    def test_put(self):
        df = DataFrame('test')
        t = DfWriter(df, 10)
        t.start()
        t.join()
        self.assertEqual(df.name, 'test')
        self.assertEqual(df.size, 10)
        datas = df.get(0, 4)
        self.assertEqual(len(datas), 4)

        datas = df.get(8, 4)
        self.assertEqual(datas, None)

        self.assertFalse(df.sealed)
        df.seal()
        self.assertTrue(df.sealed)

        datas = df.get(8, 4)
        self.assertEqual(len(datas), 2)

    def test_multithread(self):
        df = DataFrame('test')
        data_size = 10
        t = DfWriter(df, data_size)
        t.set_sealed_when_stop()
        t.start()
        q = queue.Queue()

        def read(df: DataFrame, q: queue.Queue):
            index = 0
            while True:
                items = df.get(index, 2)
                if items:
                    for item in items:
                        q.put(item)
                        index += 1
                if df.sealed:
                    break

        runner = MultiThreadRunner(target=read, args=(df, q), thread_num=10)

        runner.start()
        runner.join()

        self.assertEqual(q.qsize(), data_size * 10)

    def test_multithread_block_get(self):
        df = DataFrame('test')
        data_size = 10
        t = DfWriter(df, data_size)
        t.set_sealed_when_stop()
        t.start()
        q = queue.Queue()

        def read(df: DataFrame, q: queue.Queue):
            index = 0
            while True:
                items = df.get(index, 2, True)
                if items:
                    for item in items:
                        q.put(item)
                        index += 1
                    continue
                if df.sealed:
                    break

        runner = MultiThreadRunner(target=read, args=(df, q), thread_num=10)

        runner.start()
        runner.join()

        self.assertEqual(q.qsize(), data_size * 10)

    def test_notify(self):
        df = DataFrame('test')

        def read(df: DataFrame):
            index = 0
            while True:
                item = df.get(index, 2, True)
                assert item is None
                break

        runner = MultiThreadRunner(target=read, args=(df, ), thread_num=10)
        runner.start()
        time.sleep(0.02)

        # if not call notify_block_readers, reader thread will never return
        df.notify_block_readers()
        runner.join()


class TestMapIterator(unittest.TestCase):
    """
    map iterator basic test
    """

    def test_map_iterator(self):
        df = DataFrame('test')
        it = df.map_iter()
        data_size = 100
        t = DfWriter(df, data_size)
        t.start()
        t.set_sealed_when_stop()
        q = queue.Queue()

        def read(it: DataFrameIterator, q: queue.Queue):
            for item in it:
                if item is not None:
                    q.put(item)
                time.sleep(0.01)

        runner = MultiThreadRunner(target=read, args=(it, q), thread_num=1)
        runner.start()
        runner.join()
        self.assertEqual(q.qsize(), 100)

    def test_block_map_iterator(self):
        df = DataFrame('test')
        it = df.map_iter(True)
        data_size = 100
        t = DfWriter(df, data_size)
        t.start()
        t.set_sealed_when_stop()
        q = queue.Queue()

        def read(it: DataFrameIterator, q: queue.Queue):
            for item in it:
                if item is not None:
                    q.put(item)
                time.sleep(0.01)

        runner = MultiThreadRunner(target=read, args=(it, q), thread_num=1)
        runner.start()
        runner.join()
        self.assertEqual(q.qsize(), 100)

    def test_notify(self):
        df = DataFrame('test')
        it = df.map_iter()

        def read(it: DataFrameIterator):
            for item in it:
                assert item is None
                break

        runner = MultiThreadRunner(target=read, args=(it, ), thread_num=5)
        runner.start()
        time.sleep(0.5)
        it.notify()
        runner.join()


if __name__ == '__main__':
    unittest.main()
