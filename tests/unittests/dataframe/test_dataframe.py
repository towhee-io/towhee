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


from pprint import pprint
import unittest
import time
import queue

from towhee.dataframe import DataFrame, DataFrameIterator

from tests.unittests.test_util.dataframe_test_util import DfWriter, MultiThreadRunner


class TestDataframe(unittest.TestCase):
    """
    Test dataframe basic function
    """

    def test_put(self):
        df = DataFrame('test', [('letter', 'str'), ('num', 'int')])
        data = ('a', 1)
        t = DfWriter(df, 10, data = data)
        t.start()
        t.join()
        self.assertEqual(df.name, 'test')
        self.assertEqual(df.size, 10)
        datas = df.get(0, 4)
        self.assertEqual(len(datas[1]), 4)

        datas = df.get(8, 4)
        self.assertEqual(datas[1], None)

        self.assertFalse(df.sealed)
        df.seal()
        self.assertTrue(df.sealed)

        datas = df.get(8, 4)
        self.assertEqual(len(datas), 2)

    def test_multithread(self):
        df = DataFrame('test', [('letter', 'str'), ('num', 'int')])
        data = ('a', 1)
        data_size = 10
        t = DfWriter(df, data_size, data = data)
        t.set_sealed_when_stop()
        t.start()
        q = queue.Queue()

        def read(df: DataFrame, q: queue.Queue):
            index = 0
            while True:
                items = df.get(index, 2)
                if items[1]:
                    for item in items[1]:
                        q.put(item)
                        index += 1
                if df.sealed:
                    items = df.get(index, 100)
                    if items[1]:
                        for item in items[1]:
                            q.put(item)
                    break

        runner = MultiThreadRunner(target=read, args=(df, q), thread_num=10)

        runner.start()
        runner.join()

        self.assertEqual(q.qsize(), data_size * 10)

    def test_multithread_block_get(self):
        df = DataFrame('test', [('letter', 'str'), ('num', 'int')])
        data = ('a', 1)
        data_size = 10
        t = DfWriter(df, data_size, data = data)
        t.set_sealed_when_stop()
        t.start()
        q = queue.Queue()

        def read(df: DataFrame, q: queue.Queue):
            index = 0
            while True:
                items = df.get(index, 2)
                if items[1]:
                    for item in items[1]:
                        q.put(item)
                        index += 1
                    continue
                if df.sealed:
                    items = df.get(index, 100)
                    if items[1]:
                        for item in items[1]:
                            q.put(item)
                    break

        runner = MultiThreadRunner(target=read, args=(df, q), thread_num=10)

        runner.start()
        runner.join()

        self.assertEqual(q.qsize(), data_size * 10)

    def test_notify(self):
        df = DataFrame('test', [('letter', 'str'), ('num', 'int')])
        data = ('a', 1)
        data_size = 10

        def read(df: DataFrame):
            index = 0
            while True:
                item = df.get(index, 2)
                assert item[1] is None
                break

        runner = MultiThreadRunner(target=read, args=(df, ), thread_num=10)
        runner.start()
        time.sleep(0.02)

        # if not call notify_block_readers, reader thread will never return
        df.unblock_all()
        runner.join()


if __name__ == '__main__':
    unittest.main()
