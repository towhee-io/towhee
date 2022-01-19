import unittest
import time
import threading
import queue

from towhee.dataframe.dataframe_v2 import DataFrame
from towhee.dataframe.iterators import MapIterator, BatchIterator
from towhee.types._frame import _Frame


class TestIterators(unittest.TestCase):
    """Basic test case for `TestIterators`.
    """
    def remove_frame(self, inputs):
            if not isinstance(inputs, tuple):
                temp = []
                for x in inputs:
                    temp.append(self.remove_frame(x))
                return temp
            else:
                return tuple(inputs[1:])

    def test_map_iters(self):
        data = [(_Frame(0), 0, 'a'), (_Frame(1), 1, 'b'), (_Frame(2), 2, 'c'), (_Frame(3), 3, 'd'), (_Frame(4), 4, 'e'), (_Frame(5), 5, 'a'), (_Frame(6), 6, 'b'),
            (_Frame(7), 7, 'c'), (_Frame(8), 8, 'd'), (_Frame(9), 9, 'e'),]

        columns = [('_frame', _Frame), ('digit', int), ('letter', str)]
        df = DataFrame(columns, name = 'my_df', data = data)
        df.seal()

        it = MapIterator(df, block=False)
        it2 = BatchIterator(df, batch_size=4, step= 4, block=False)

        

        # MapIter
        res = []
        for row in it:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')]])

        # BatchIter
        res = []
        for rows in it2:
            res.append(rows)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e')]])

        # Assert cleared df
        self.assertEqual(df.physical_size, 0)

    def test_blocking_map_iters(self):
        data = data = [(_Frame(0), 0, 'a'), (_Frame(1), 1, 'b'), (_Frame(2), 2, 'c'), (_Frame(3), 3, 'd'), (_Frame(4), 4, 'e'), (_Frame(5), 5, 'a'), (_Frame(6), 6, 'b'),
            (_Frame(7), 7, 'c'), (_Frame(8), 8, 'd'), (_Frame(9), 9, 'e'),]

        columns = [('_frame', _Frame), ('digit', int), ('letter', str)]
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
        df.put((_Frame(10), 10, 'f'))
        df.put((_Frame(11), 11, 'g'))
        time.sleep(.1)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')],
            [(10, 'f')], [(11, 'g')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e'), (10, 'f'),
            (11, 'g')]])

        self.assertEqual(df.physical_size, 0)
        time.sleep(.1)
        df.put((_Frame(12), 12, 'h'))
        df.put((_Frame(13), 13, 'i'))
        time.sleep(.1)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')],
            [(10, 'f')], [(11, 'g')], [(12, 'h')], [(13, 'i')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e'), (10, 'f'),
            (11, 'g')]])

        self.assertEqual(df.physical_size, 2)
        time.sleep(.1)
        df.seal()
        time.sleep(.1)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')],
            [(10, 'f')], [(11, 'g')], [(12, 'h')], [(13, 'i')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e'), (10, 'f'),
            (11, 'g')], [(12, 'h'), (13, 'i')]])

        self.assertEqual(df.physical_size, 0)
        x.join()
        x2.join()

    def test_kill_iters(self):
        data = [(_Frame(0), 0, 'a'), (_Frame(1), 1, 'b'), (_Frame(2), 2, 'c'), (_Frame(3), 3, 'd'), (_Frame(4), 4, 'e'), (_Frame(5), 5, 'a'), (_Frame(6), 6, 'b'),
            (_Frame(7), 7, 'c'), (_Frame(8), 8, 'd'), (_Frame(9), 9, 'e'),]

        columns = [('_frame', _Frame), ('digit', int), ('letter', str)]
        df = DataFrame(columns, name = 'my_df', data = data)

        q = queue.Queue()
        q2 = queue.Queue()

        it = MapIterator(df, block=True)
        it2 = BatchIterator(df, batch_size=4, step= 4, block=True)

        def read(iterator, q: queue.Queue):
            for x in iterator:
                q.put(x)

        x = threading.Thread(target=read, args=(it,q,))
        x2 = threading.Thread(target=read, args=(it2,q2,))

        x.start()
        x2.start()

        df.unblock_iters()

        x.join()
        x2.join()

        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')]])

        self.assertEqual([value for _, value in df.iterators.items()], [float('inf'), float('inf')])
        
if __name__ == '__main__':
    unittest.main()
