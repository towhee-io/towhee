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
import threading
import queue


from towhee.array import Array
from towhee.dataframe.dataframe_v2 import DataFrame
from towhee.dataframe.iterators import MapIterator, BatchIterator

from towhee.tests.test_util.dataframe_test_util import DfWriter, MultiThreadRunner



class TestDataframe(unittest.TestCase):
    """
    Test dataframe basic function
    """

    def get_columns(self):
            return [('digit', int), ('letter', str)]

    def get_tuples(self):
        return [(0, 'a'), (1, 'b'), (2, 'c')]

    def get_arrays(self):
        return [Array([0, 1, 2]), Array(['a', 'b', 'c'])]

    def get_dict(self):
        return {('digit', int): Array([0, 1, 2]), ('letter', str): Array(['a', 'b', 'c'])}


    # def test_constructors(self):

    #     def check_data(df):
    #         for i in range(3):
    #             self.assertEqual(df['digit'][i], i)
    #             self.assertEqual(df['letter'][i], chr(ord('a') + i))
    #             self.assertEqual(df[i][0], i)
    #             self.assertEqual(df[i][1], chr(ord('a') + i))
    #         for i, row in enumerate(df.iter()):
    #             self.assertEqual(row[0], i)
    #             self.assertEqual(row[1], chr(ord('a') + i))

    #     # from list[tuple]
    #     data = self.get_tuples()
    #     columns = self.get_columns()
    #     df = DataFrame(columns, name = 'my_df', data = data)
    #     df1 = DataFrame(columns, name = 'my_df')
    #     print(df1)
    #     df1.put([1, 'a'])
    #     print(df1)

    #     df.put([3,'d'])
    #     df.put((4, 'e'))
    #     df.put({'digit': 5, 'letter': 'f'})
    #     df.seal()
    #     # MapIters
    #     it = MapIterator(df, block=True)
    #     it2 = MapIterator(df)

    #     for i, row in enumerate(it):
    #         print('row, (values, ): ', i, row)
    #         print('\nDF Values:\n')
    #         print(df)

    #     for i, row in enumerate(it2):
    #         print('row, (values, ): ', i, row)
    #         print('\nDF Values:\n')
    #         print(df)
    
    #     # BatchIter
    #     it = BatchIterator(df, batch_size=4, block=True)
    #     for i, rows in enumerate(it):
    #         print('row, (values, ): ', i, rows)
    #         print('\nDF Values:\n')
    #         print(df)

        
       
    #     check_data(df)
        
    #     # from list[towhee.Array]
    #     data = get_arrays()
    #     columns = get_columns()
    #     df = DataFrame(columns, name = 'my_df', data = data)
    #     df.seal()
    #     check_data(df)

    #     # from dict[str, towhee.Array]
    #     data = get_dict()
    #     df = DataFrame(None, name = 'my_df', data = data)
    #     print(df)

    #     df.seal()
    #     check_data(df)
    #     for x in range(len(df)):

    # def test_put(self):
    #     columns = self.get_columns()
    #     data = (0, 'a')
    #     df = DataFrame(columns, name = 'test')
    #     t = DfWriter(df, 10, data)
    #     t.start()
    #     t.join()
    #     self.assertEqual(df.name, 'test')
    #     self.assertEqual(len(df), 10)
    #     datas = df.get(0, 4)
    #     self.assertEqual(datas[0], 'Approved_Continue')
    #     self.assertEqual(len(datas[1]), 4)

    #     datas = df.get(8, 4)
    #     self.assertEqual(datas[0], 'Index_OOB_Unsealed')
    #     self.assertEqual(datas[1], None)


    #     self.assertFalse(df.sealed)
    #     df.seal()
    #     self.assertTrue(df.sealed)

    #     datas = df.get(8, 4)
    #     print(datas)
    #     self.assertEqual(datas[0], 'Approved_Done')
    #     self.assertEqual(len(datas[1]), 2)

    # def test_multithread(self):
    #     columns = self.get_columns()
    #     df = DataFrame(columns, name = 'test')
    #     data_size = 10
    #     data = (0, 'a')
    #     t = DfWriter(df, data_size, data=data)
    #     t.set_sealed_when_stop()
    #     t.start()
    #     q = queue.Queue()

    #     def read(df: DataFrame, q: queue.Queue):
    #         index = 0
    #         while True:
    #             items = df.get(index, 2)
    #             if items[1]:
    #                 for item in items[1]:
    #                     q.put(item)
    #                     index += 1
    #             if df.sealed:
    #                 items = df.get(index, 100)
    #                 if items[1]:
    #                     for item in items[1]:
    #                         q.put(item)
    #                 break

    #     runner = MultiThreadRunner(target=read, args=(df, q), thread_num=10)

    #     runner.start()
    #     runner.join()

    #     self.assertEqual(q.qsize(), data_size * 10)







if __name__ == '__main__':
    unittest.main()
