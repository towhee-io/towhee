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

from towhee.dataframe.array import Array
from towhee.dataframe.dataframe_v2 import DataFrame, Responses
from tests.unittests.test_util.dataframe_test_util import DfWriter, MultiThreadRunner


class TestDataframe(unittest.TestCase):
    """Basic test case for `DataFrame`.
    """

    def get_columns(self):
        return [('digit', int), ('letter', str)]

    def get_tuples(self):
        return [(0, 'a'), (1, 'b'), (2, 'c')]

    def get_arrays(self):
        return [Array([0, 1, 2]), Array(['a', 'b', 'c'])]

    def get_dict(self):
        return {'Col_0': Array(name = 'digit', data=[0, 1, 2]), 'Col_1': Array(name='letter', data = ['a', 'b', 'c'])}


    def test_constructors(self):

        def check_data(df):
            for i in range(3):
                self.assertEqual(df['digit'][i], i)
                self.assertEqual(df['letter'][i], chr(ord('a') + i))
                self.assertEqual(df[i][0], i)
                self.assertEqual(df[i][1], chr(ord('a') + i))

        def check_data_default(df):
            for i in range(3):
                self.assertEqual(df['Col_0'][i], i)
                self.assertEqual(df['Col_1'][i], chr(ord('a') + i))
                self.assertEqual(df[i][0], i)
                self.assertEqual(df[i][1], chr(ord('a') + i))

        # from list[tuple] with cols
        data = self.get_tuples()
        columns = self.get_columns()
        df = DataFrame(columns, name = 'my_df', data = data)
        df.seal()
        check_data(df)

        # from list[tuple] without cols
        data = self.get_tuples()
        df = DataFrame(None, name = 'my_df', data = data)
        df.seal()
        check_data_default(df)

        # from list[towhee.dataframe.Array] with cols
        data = self.get_arrays()
        columns = self.get_columns()
        df = DataFrame(columns, name = 'my_df', data = data)
        df.seal()
        check_data(df)

         # from list[towhee.dataframe.Array] without cols
        data = self.get_arrays()
        df = DataFrame(None, name = 'my_df', data = data)
        df.seal()
        check_data_default(df)

        # from dict[str, towhee.dataframe.Array] with cols.
        data = self.get_dict()
        columns = self.get_columns()
        df = DataFrame(columns, name = 'my_df', data = data)
        df.seal()
        check_data(df)

        # from dict[str, towhee.dataframe.Array] without cols.
        data = self.get_dict()
        df = DataFrame(None, name = 'my_df', data = data)
        df.seal()
        check_data_default(df)

        self.assertEqual(df.name, 'my_df')


    def test_put(self):
        columns = self.get_columns()
        data = (0, 'a')
        df = DataFrame(columns, name = 'test')
        t = DfWriter(df, 10, data)
        t.start()
        t.join()
        self.assertEqual(df.name, 'test')
        self.assertEqual(len(df), 10)
        datas = df.get(0, 4)
        self.assertEqual(datas[0], Responses.APPROVED_CONTINUE)
        self.assertEqual(len(datas[1]), 4)

        datas = df.get(8, 4)
        self.assertEqual(datas[0], Responses.INDEX_OOB_UNSEALED)
        self.assertEqual(datas[1], None)


        self.assertFalse(df.sealed)
        df.seal()
        self.assertTrue(df.sealed)

        datas = df.get(8, 4)
        self.assertEqual(datas[0], Responses.APPROVED_DONE)
        self.assertEqual(len(datas[1]), 2)

    def test_multithread(self):
        columns = self.get_columns()
        df = DataFrame(columns, name = 'test')
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


if __name__ == '__main__':
    unittest.main()
