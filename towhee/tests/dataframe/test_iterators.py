import unittest
import time
import threading
import queue

from towhee.dataframe.dataframe_v2 import DataFrame
from towhee.dataframe.iterators import MapIterator, BatchIterator


class TestIterators(unittest.TestCase):
    """Basic test case for `TestIterators`.
    """
    def test_iters(self):
        data = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'a'), (6, 'b'), (7, 'c'), (8, 'd'), (9, 'e'),]
        columns = [('digit', int), ('letter', str)]
        df = DataFrame(columns, name = 'my_df', data = data)
        df.seal()

        it = MapIterator(df, block=False)
        it2 = BatchIterator(df, batch_size=4, step= 4, block=False)

        # MapIters
        res = []
        for row in it:
            res.append(row)
        self.assertEqual(res, [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')]])

        # BatchIter
        res = []
        for rows in it2:
            res.append(rows)
        self.assertEqual(res, [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e')]])

        # Assert cleared df
        self.assertEqual(df.physical_size, 0)

    def test_blocking_iters(self):
        data = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'a'), (6, 'b'), (7, 'c'), (8, 'd'), (9, 'e'),]
        columns = [('digit', int), ('letter', str)]
        df = DataFrame(columns, name = 'my_df', data = data)
        q = queue.Queue()
        it = MapIterator(df, block=True)
        q2 = queue.Queue()
        it2 = BatchIterator(df, batch_size=4, step= 4, block=True)

        def read(iterator, q: queue.Queue):
            for x in iterator:
                q.put(x)

        x = threading.Thread(target=read, args=(it,q,))
        x.start()
        x2 = threading.Thread(target=read, args=(it2,q2,))
        x2.start()
        time.sleep(.1)
        df.put((10, 'f'))
        df.put((11, 'g'))
        time.sleep(.1)
        self.assertEqual(list(q.queue), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')], [(10, 'f')], [(11, 'g')]])
        self.assertEqual(list(q2.queue), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e'), (10, 'f'), (11, 'g')]])
        self.assertEqual(df.physical_size, 0)
        time.sleep(.1)
        df.put((12, 'h'))
        df.put((13, 'i'))
        time.sleep(.1)
        self.assertEqual(list(q.queue), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')], [(10, 'f')], [(11, 'g')], [(12, 'h')], [(13, 'i')]])
        self.assertEqual(list(q2.queue), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e'), (10, 'f'), (11, 'g')]])
        self.assertEqual(df.physical_size, 2)
        time.sleep(.1)
        df.seal()
        time.sleep(.1)
        self.assertEqual(list(q.queue), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')], [(10, 'f')], [(11, 'g')], [(12, 'h')], [(13, 'i')]])
        self.assertEqual(list(q2.queue), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e'), (10, 'f'), (11, 'g')], [(12, 'h'), (13, 'i')]])
        self.assertEqual(df.physical_size, 0)
        x.join()
        x2.join()

if __name__ == '__main__':
    unittest.main()
