import unittest
import time
import threading
import queue

from towhee.dataframe.dataframe_v2 import DataFrame
from towhee.dataframe.iterators import MapIterator, BatchIterator, WindowIterator
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
            values = list(inputs)
            values = [value for value in values if not isinstance(value, _Frame)]
            return tuple(values)

    def gen_data(self):
        data = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'a'), (6, 'b'),
            (7, 'c'), (8, 'd'), (9, 'e'),]
        cols = [('digit', int), ('letter', str)]
        return data, cols

    def test_map_iters(self):
        data, columns = self.gen_data()
        df = DataFrame(columns, name = 'my_df', data = data)
        df.seal()
        it = MapIterator(df)
        it2 = BatchIterator(df, batch_size=4, step= 4)
        print(df)

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
        data, columns = self.gen_data()
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
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')],
            [(10, 'f')], [(11, 'g')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e'), (10, 'f'),
            (11, 'g')]])

        self.assertEqual(df.physical_size, 0)
        time.sleep(.1)
        df.put((12, 'h'))
        df.put((13, 'i'))
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


    def test_window_iters(self):
        data, columns = self.gen_data()
        df = DataFrame(columns, name = 'my_df', data = data)
        df.seal()

        it = WindowIterator(df, window_size = 1)
        it2 = WindowIterator(df, window_size = 2)
        it3 = WindowIterator(df, window_size = 3)


        # WindowIter window of 2
        res = []
        for row in it:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')]])

        # WindowIter window of 2
        res = []
        for row in it2:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b')], [(2, 'c'), (3, 'd')], [(4, 'e'),
            (5, 'a')], [(6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e')]])

        # WindowIter window of 3
        res = []
        for row in it3:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b'), (2, 'c')], [(3, 'd'), (4, 'e'),
            (5, 'a')], [(6, 'b'), (7, 'c'), (8, 'd')], [(9, 'e')]])

        # Assert cleared df
        self.assertEqual(df.physical_size, 0)

    def test_blocking_window_iters(self):
        data, columns = self.gen_data()
        df = DataFrame(columns, name = 'my_df', data = data)
        q = queue.Queue()
        it = WindowIterator(df, window_size = 2, block=True)
        q2 = queue.Queue()
        it2 = WindowIterator(df, window_size = 3, block=True)

        def read(iterator, q: queue.Queue):
            for x in iterator:
                q.put(x)

        x = threading.Thread(target=read, args=(it,q,))
        x.start()
        x2 = threading.Thread(target=read, args=(it2,q2,))
        x2.start()
        df.put((10, 'f'))
        df.put((11, 'g'))
        time.sleep(.1)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a'), (1, 'b')],
            [(2, 'c'), (3, 'd')], [(4, 'e'), (5, 'a')], [(6, 'b'), (7, 'c')], [(8, 'd'),
            (9, 'e')]])
        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c')],
            [(3, 'd'), (4, 'e'), (5, 'a')], [(6, 'b'), (7, 'c'), (8, 'd')]])
        self.assertEqual(df.physical_size, 3)
        df.put((12, 'h'))
        df.put((13, 'i'))
        time.sleep(.1)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a'), (1, 'b')], [(2, 'c'),
            (3, 'd')], [(4, 'e'), (5, 'a')], [(6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e')],
            [(10, 'f'),(11, 'g')]])
        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c')],
            [(3, 'd'), (4, 'e'), (5, 'a')], [(6, 'b'), (7, 'c'), (8, 'd')], [(9, 'e'), (10, 'f'),
            (11, 'g')]])
        self.assertEqual(df.physical_size, 2)
        df.seal()
        time.sleep(.1)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a'), (1, 'b')], [(2, 'c'),
            (3, 'd')], [(4, 'e'), (5, 'a')], [(6, 'b'), (7, 'c')], [(8, 'd'), (9, 'e')],
            [(10, 'f'), (11, 'g')], [(12, 'h'), (13, 'i')]])
        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c')],
            [(3, 'd'), (4, 'e'), (5, 'a')], [(6, 'b'), (7, 'c'), (8, 'd')], [(9, 'e'),
            (10, 'f'), (11, 'g')], [(12, 'h'), (13, 'i')]])
        self.assertEqual(df.physical_size, 0)
        x.join()
        x2.join()

    def test_kill__iters(self):
        data, columns = self.gen_data()
        df = DataFrame(columns, name = 'my_df', data = data)

        q = queue.Queue()
        q2 = queue.Queue()
        q3 = queue.Queue()

        it = MapIterator(df, block = True)
        it2 = BatchIterator(df, batch_size = 4, step= 4, block = True)
        it3 = WindowIterator(df, window_size = 3, block = True)

        def read(iterator, q: queue.Queue):
            for x in iterator:
                q.put(x)

        x = threading.Thread(target = read, args = (it,q,))
        x2 = threading.Thread(target = read, args = (it2,q2,))
        x3 = threading.Thread(target = read, args = (it3, q3))

        x.start()
        x2.start()
        x3.start()

        df.unblock_iters()

        x.join()
        x2.join()
        x3.join()

        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'a')], [(6, 'b')], [(7, 'c')], [(8, 'd')], [(9, 'e')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'a'), (6, 'b'), (7, 'c')]])

        self.assertEqual(self.remove_frame(list(q3.queue)), [[(0, 'a'), (1, 'b'), (2, 'c')],
            [(3, 'd'), (4, 'e'), (5, 'a')], [(6, 'b'), (7, 'c'), (8, 'd')]])

        self.assertEqual([value for _, value in df.iterators.items()], [float('inf'), float('inf'), float('inf')])



if __name__ == '__main__':
    unittest.main()
