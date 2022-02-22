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
import queue

from towhee.types._frame import _Frame, FRAME
from towhee.dataframe.dataframe import DataFrame
from tests.unittests.test_util.dataframe_test_util import DfWriter, MultiThreadRunner


class TestDataframe(unittest.TestCase):
    """Basic test case for `DataFrame`.
    """

    def get_columns(self):
        return [('digit', 'int'), ('letter', 'str')]

    def get_tuples(self, frames = False):
        if frames:
            return [(0, 'a', _Frame(row_id = 4, timestamp=4)), (1, 'b', _Frame(5, timestamp=5)), (2, 'c', _Frame(6, timestamp=6))]
        return [(0, 'a'), (1, 'b'), (2, 'c')]

    def get_dict(self, frames = False):
        if frames:
            ret = []
            ret.append({'digit': 0, 'letter': 'a', FRAME: _Frame(row_id = 4, timestamp=4)})
            ret.append({'digit': 1, 'letter': 'b', FRAME: _Frame(row_id = 5, timestamp=5)})
            ret.append({'digit': 2, 'letter': 'c', FRAME: _Frame(row_id = 6, timestamp=6)})
            return ret
        ret = []
        ret.append({'digit': 0, 'letter': 'a'})
        ret.append({'digit': 1, 'letter': 'b'})
        ret.append({'digit': 2, 'letter': 'c'})
        return ret

    def test_constructors(self):

        columns = self.get_columns()
        df = DataFrame('my_df', columns)
        df.seal()
        self.assertEqual(df.name, 'my_df')
        self.assertEqual(df.sealed, True)
        self.assertEqual(len(df), 0)

    def test_put_tuple(self):

        #  Testing put tuple with no frame col in both data and cols.
        columns = self.get_columns()
        data = self.get_tuples(frames = False)
        df = DataFrame('test', columns)
        df.put(data)
        count = 0
        for x in df:
            self.assertEqual(x[0], count)
            self.assertEqual(x[2].row_id, count)
            count += 1

        #  Testing put tuple with frame col in both data and cols.
        columns = self.get_columns()
        data = self.get_tuples(frames = True)
        df = DataFrame('test', columns)
        df.put(data)
        count = 0
        for x in df:
            self.assertEqual(x[0], count)
            self.assertEqual(x[2].row_id, count)
            count += 1

        #  Testing put tuple with no frame col in cols and with frame cols in data.
        columns = self.get_columns()
        data = self.get_tuples(frames = True)
        df = DataFrame('test', columns)
        df.put(data)
        count = 0
        for x in df:
            self.assertEqual(x[0], count)
            self.assertEqual(x[2].row_id, count)
            count += 1

        #  Testing single tuple put and df accesses.
        data = (0, 'a')
        columns = self.get_columns()
        df = DataFrame('test', columns)
        t = DfWriter(df, 10, data)
        t.start()
        t.join()
        self.assertEqual(df.name, 'test')
        self.assertEqual(len(df), 10)
        datas = df.get(0, 4)
        self.assertEqual(len(datas), 4)
        datas = df.get(8, 4)
        self.assertEqual(datas, None)
        self.assertFalse(df.sealed)
        df.seal()
        self.assertTrue(df.sealed)
        datas = df.get(8, 4)
        self.assertEqual(len(datas), 2)

    def test_put_dict(self):
        #  Testing put dict with no frame col in both data and cols.
        columns = self.get_columns()
        data = self.get_dict(frames = False)
        df = DataFrame('test', columns)
        df.put(data)
        count = 0
        for x in df:
            self.assertEqual(x[0], count)
            self.assertEqual(x[2].row_id, count)
            count += 1

        #  Testing put dict with frame col in both data and cols.
        columns = self.get_columns()
        data = self.get_dict(frames = True)
        df = DataFrame('test', columns)
        df.put(data)
        count = 0
        for x in df:
            self.assertEqual(x[0], count)
            self.assertEqual(x[2].row_id, count)
            count += 1

        #  Testing put dict with no frame col in cols and with frame cols in data.
        columns = self.get_columns()
        data = self.get_tuples(frames = True)
        df = DataFrame('test', columns)
        df.put(data)
        count = 0
        for x in df:
            self.assertEqual(x[0], count)
            self.assertEqual(x[2].row_id, count)
            count += 1

        #  Testing single dict put and df accesses.
        data = {'digit': 0, 'letter': 'a'}
        columns = self.get_columns()
        df = DataFrame('test', columns)
        t = DfWriter(df, 10, data)
        t.start()
        t.join()
        self.assertEqual(df.name, 'test')
        self.assertEqual(len(df), 10)
        datas = df.get(0, 4)
        self.assertEqual(datas[0][0], 0)
        self.assertEqual(len(datas), 4)

        datas = df.get(8, 4)
        self.assertEqual(datas, None)


        self.assertFalse(df.sealed)
        df.seal()
        self.assertTrue(df.sealed)

        datas = df.get(8, 4)
        self.assertEqual(len(datas), 2)

    def test_multithread(self):
        columns = self.get_columns()
        df = DataFrame('test', columns)
        data_size = 10
        data = (0, 'a')
        t = DfWriter(df, data_size, data=data)
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
                    items = df.get(index, 100)
                    if items:
                        for item in items:
                            q.put(item)
                    break

        runner = MultiThreadRunner(target=read, args=(df, q), thread_num=10)

        runner.start()
        runner.join()

        self.assertEqual(q.qsize(), data_size * 10)


if __name__ == '__main__':
    unittest.main()
