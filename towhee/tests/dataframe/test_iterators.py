import unittest
import time
import threading
import queue


from towhee.array import Array
from towhee.dataframe.dataframe_v2 import DataFrame
from towhee.dataframe.iterators import MapIterator, BatchIterator

from towhee.tests.test_util.dataframe_test_util import DfWriter, MultiThreadRunner



class TestDataframe(unittest.TestCase):
    def test_iters(self):
        data = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'a'), (6, 'b'), (7, 'c'), (8, 'd'), (9, 'e'),]
        columns = [('digit', int), ('letter', str)]
        df = DataFrame(columns, name = 'my_df', data = data)
        df.seal()
        
        it = MapIterator(df, block=False)
        it2 = BatchIterator(df, batch_size=4, step= 4, block=False)
        # MapIters
        res = []
        for i, row in enumerate(it):
            # print('row, (values, ): ', i, row)
            # print('\nDF Values:\n')
            # print(df)
            res.append(row)
        self.A
            

        # BatchIter
        for i, rows in enumerate(it2):
            print('batch, (values, ): ', i, rows)
            print('\nDF Values:\n')
            print(df)
        
    # def test_blocking_iters(self):
        

if __name__ == '__main__':
    unittest.main()