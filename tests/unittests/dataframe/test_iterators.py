import unittest
import time
import threading
import queue

from string import ascii_lowercase

from towhee.dataframe.dataframe import DataFrame
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

    def gen_data(self, start, end):
        data = []
        for x in range(start, end):
            letter = x%25
            temp_data = (x, ascii_lowercase[letter], _Frame(row_id = x, timestamp = x + 10**10))
            data.append(temp_data)
        cols = [('digit', 'int'), ('letter', 'str')]
        return data, cols

    def test_map_iters(self):
        data, columns = self.gen_data(0, 10)
        df = DataFrame('my_df', columns)
        df.put(data)
        df.seal()
        it = MapIterator(df)
        it2 = BatchIterator(df, batch_size=4, step= 4)

        # MapIter
        res = []
        for row in it:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'f')], [(6, 'g')], [(7, 'h')], [(8, 'i')], [(9, 'j')]])

        # BatchIter
        res = []
        for rows in it2:
            res.append(rows)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h')],
            [(8, 'i'), (9, 'j')]])

        # Assert cleared df
        df.gc()
        self.assertEqual(len(df), 0)

    def test_blocking_map_iters(self):
        data, columns = self.gen_data(0, 10)
        df = DataFrame('my_df', columns)
        df.put(data)
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
        time.sleep(.5)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'f')], [(6, 'g')], [(7, 'h')], [(8, 'i')], [(9, 'j')],
            [(10, 'f')], [(11, 'g')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h')],
            [(8, 'i'), (9, 'j'), (10, 'f'), (11, 'g')]])

        df.gc()
        self.assertEqual(len(df), 0)
        time.sleep(.1)
        df.put((12, 'h'))
        df.put((13, 'i'))
        time.sleep(.1)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'f')], [(6, 'g')], [(7, 'h')], [(8, 'i')], [(9, 'j')],
            [(10, 'f')], [(11, 'g')], [(12, 'h')], [(13, 'i')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h')], [(8, 'i'), (9, 'j'), (10, 'f'),
            (11, 'g')]])

        df.gc()
        self.assertEqual(len(df), 2)
        time.sleep(.1)
        df.seal()
        time.sleep(.1)
        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'f')], [(6, 'g')], [(7, 'h')], [(8, 'i')], [(9, 'j')],
            [(10, 'f')], [(11, 'g')], [(12, 'h')], [(13, 'i')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h')], [(8, 'i'), (9, 'j'), (10, 'f'),
            (11, 'g')], [(12, 'h'), (13, 'i')]])

        df.gc()
        self.assertEqual(len(df), 0)
        x.join()
        x2.join()




    def test_window_iters(self):
        data, columns = self.gen_data(0, 10)
        df = DataFrame('my_df', columns)
        df.put(data)
        df.seal()

        it = WindowIterator(df, window_size = 1, use_timestamp = False)
        it2 = WindowIterator(df, window_size = 2, use_timestamp = False)
        it3 = WindowIterator(df, window_size = 3, use_timestamp = False)

        it4 = WindowIterator(df, window_size = 4, step = 2, use_timestamp = False)
        it5 = WindowIterator(df, window_size = 5, step = 3, use_timestamp = False)


        # WindowIter window of 1
        res = []
        for row in it:
            res.append(row)
        res = self.remove_frame(res)

        self.assertEqual(res, [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'f')], [(6, 'g')], [(7, 'h')], [(8, 'i')], [(9, 'j')]])

        # WindowIter window of 2
        res = []
        for row in it2:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b')], [(2, 'c'), (3, 'd')], [(4, 'e'),
            (5, 'f')], [(6, 'g'), (7, 'h')], [(8, 'i'), (9, 'j')]])

        # WindowIter window of 3
        res = []
        for row in it3:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b'), (2, 'c')], [(3, 'd'), (4, 'e'),
            (5, 'f')], [(6, 'g'), (7, 'h'), (8, 'i')], [(9, 'j')]])

        # WindowIter window of 4, step = 2
        res = []
        for row in it4:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(2, 'c'), (3, 'd'), (4, 'e'), (5, 'f')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h')],
            [(6, 'g'), (7, 'h'), (8, 'i'), (9, 'j')],
            [(8, 'i'), (9, 'j')]])

        # WindowIter window of 5, step = 3
        res = []
        for row in it5:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')],
            [(3, 'd'), (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h')],
            [(6, 'g'), (7, 'h'), (8, 'i'), (9, 'j')],
            [(9, 'j')]])

        # Assert cleared df
        df.gc()
        self.assertEqual(len(df), 0)

    def test_step_window_timestamp_iters(self):
        data, columns = self.gen_data(0, 12)

        df = DataFrame('my_df', columns)
        df.put(data)
        df.seal()

        it1 = WindowIterator(df, window_size = 10, step = 5, use_timestamp=False)
        it2 = WindowIterator(df, window_size = 2, step = 5, use_timestamp=False)


        res = []
        for row in it1:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'), (8, 'i'), (9, 'j')],
            [(5, 'f'), (6, 'g'), (7, 'h'), (8, 'i'), (9, 'j'), (10, 'k'), (11, 'l')], [(10, 'k'), (11, 'l')]])

        res = []
        for row in it2:
            res.append(row)
        res = self.remove_frame(res)
        self.assertEqual(res, [[(0, 'a'), (1, 'b')], [(5, 'f'), (6, 'g')], [(10, 'k'), (11, 'l')]])
        df.gc()
        self.assertEqual(len(df), 0)

    def test_blocking_window_iters(self):
        data, columns = self.gen_data(0, 20)
        df = DataFrame('my_df', columns)
        df.put(data[:11])
        q = queue.Queue()
        it = WindowIterator(df, window_size = 2, block=True)
        q2 = queue.Queue()
        it2 = WindowIterator(df, window_size = 5, step=2, block=True)

        def read(iterator, q: queue.Queue):
            for x in iterator:
                q.put(x)

        x = threading.Thread(target=read, args=(it,q,))
        x.start()
        x2 = threading.Thread(target=read, args=(it2,q2,))
        x2.start()

        time.sleep(.1)

        self.assertEqual(self.remove_frame(list(q.queue)),
            [[(0, 'a'), (1, 'b')], [(2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'f')], [(6, 'g'), (7, 'h')],
            [(8, 'i'), (9, 'j')]])

        self.assertEqual(self.remove_frame(list(q2.queue)),
            [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')],
            [(2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'), (6, 'g')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'), (8, 'i')]])

        df.gc()
        self.assertEqual(len(df), 5)

        df.put(data[11:])

        time.sleep(.1)

        self.assertEqual(self.remove_frame(list(q.queue)),
            [[(0, 'a'), (1, 'b')],
            [(2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'f')],
            [(6, 'g'), (7, 'h')],
            [(8, 'i'), (9, 'j')],
            [(10, 'k'), (11, 'l')],
            [(12, 'm'), (13, 'n')],
            [(14, 'o'), (15, 'p')],
            [(16, 'q'), (17, 'r')]])

        self.assertEqual(self.remove_frame(list(q2.queue)),
            [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')],
            [(2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'), (6, 'g')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'), (8, 'i')],
            [(6, 'g'), (7, 'h'), (8, 'i'), (9, 'j'), (10, 'k')],
            [(8, 'i'), (9, 'j'), (10, 'k'), (11, 'l'), (12, 'm')],
            [(10, 'k'), (11, 'l'), (12, 'm'), (13, 'n'), (14, 'o')],
            [(12, 'm'), (13, 'n'), (14, 'o'), (15, 'p'), (16, 'q')],
            [(14, 'o'), (15, 'p'), (16, 'q'), (17, 'r'), (18, 's')]])

        df.gc()
        self.assertEqual(len(df), 4)

        df.seal()

        time.sleep(.1)

        self.assertEqual(self.remove_frame(list(q.queue)),
            [[(0, 'a'), (1, 'b')],
            [(2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'f')],
            [(6, 'g'), (7, 'h')],
            [(8, 'i'), (9, 'j')],
            [(10, 'k'), (11, 'l')],
            [(12, 'm'), (13, 'n')],
            [(14, 'o'), (15, 'p')],
            [(16, 'q'), (17, 'r')],
            [(18, 's'), (19, 't')]])

        self.assertEqual(self.remove_frame(list(q2.queue)),
            [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')],
            [(2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'), (6, 'g')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'), (8, 'i')],
            [(6, 'g'), (7, 'h'), (8, 'i'), (9, 'j'), (10, 'k')],
            [(8, 'i'), (9, 'j'), (10, 'k'), (11, 'l'), (12, 'm')],
            [(10, 'k'), (11, 'l'), (12, 'm'), (13, 'n'), (14, 'o')],
            [(12, 'm'), (13, 'n'), (14, 'o'), (15, 'p'), (16, 'q')],
            [(14, 'o'), (15, 'p'), (16, 'q'), (17, 'r'), (18, 's')],
            [(16, 'q'), (17, 'r'), (18, 's'), (19, 't')],
            [(18, 's'), (19, 't')]])

        df.gc()
        self.assertEqual(len(df), 0)
        x.join()
        x2.join()

    def test_kill__iters(self):
        data, columns = self.gen_data(0, 10)
        df = DataFrame('my_df', columns)
        df.put(data)

        q = queue.Queue()
        q2 = queue.Queue()
        q3 = queue.Queue()
        q4 = queue.Queue()

        it = MapIterator(df, block = True)
        it2 = BatchIterator(df, batch_size = 4, step= 4, block = True)
        it3 = WindowIterator(df, window_size = 3, block = True)
        it4 = WindowIterator(df, window_size = 5, step = 4, block = True)

        def read(iterator, q: queue.Queue):
            for x in iterator:
                q.put(x)

        x = threading.Thread(target = read, args = (it,q,))
        x2 = threading.Thread(target = read, args = (it2,q2,))
        x3 = threading.Thread(target = read, args = (it3, q3))
        x4 = threading.Thread(target = read, args = (it4, q4))

        x.start()
        x2.start()
        x3.start()
        x4.start()

        # df.unblock_iter(1)
        # print(df._map_blocked, df._window_end_blocked, df._window_start_blocked)

        # df.unblock_iter(2)
        # print(df._map_blocked, df._window_end_blocked, df._window_start_blocked)

        # df.unblock_iter(3)
        # print(df._map_blocked, df._window_end_blocked, df._window_start_blocked)

        # df.unblock_iter(4)
        # print(df._map_blocked, df._window_end_blocked, df._window_start_blocked)

        df.unblock_all()

        x.join()
        x2.join()
        x3.join()
        x4.join()

        self.assertEqual(self.remove_frame(list(q.queue)), [[(0, 'a')], [(1, 'b')], [(2, 'c')], [(3, 'd')],
            [(4, 'e')], [(5, 'f')], [(6, 'g')], [(7, 'h')], [(8, 'i')], [(9, 'j')]])

        self.assertEqual(self.remove_frame(list(q2.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h')]])

        self.assertEqual(self.remove_frame(list(q3.queue)), [[(0, 'a'), (1, 'b'), (2, 'c')],
            [(3, 'd'), (4, 'e'), (5, 'f')], [(6, 'g'), (7, 'h'), (8, 'i')]])

        self.assertEqual(self.remove_frame(list(q4.queue)), [[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')],
            [(4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'), (8, 'i')]])

        self.assertEqual([value for _, value in df.iterators.items()], [])

    def test_all_iters(self):

        def read(iterator, q: queue.Queue):
            for x in iterator:
                q.put(x)

        data, columns = self.gen_data(0, 10)
        df = DataFrame('my_df', columns)
        df.put(data)

        q = queue.Queue()
        q2 = queue.Queue()
        q3 = queue.Queue()
        q4 = queue.Queue()

        it = MapIterator(df, block = True)
        it2 = BatchIterator(df, batch_size = 4, step= 4, block = True)
        it3 = WindowIterator(df, window_size = 3, use_timestamp = True, block = True)
        it4 = WindowIterator(df, window_size = 5, step = 4, use_timestamp = False, block = True)

        x = threading.Thread(target = read, args = (it,q,))
        x2 = threading.Thread(target = read, args = (it2,q2,))
        x3 = threading.Thread(target = read, args = (it3, q3))
        x4 = threading.Thread(target = read, args = (it4, q4))

        x.start()
        x2.start()
        x3.start()
        x4.start()

        q.empty()
        q2.empty()
        q3.empty()
        q4.empty()
        data, columns = self.gen_data(10, 10234)
        df.put(data)

        time.sleep(1)
        # print(df, df.iterators)

        df.seal()

        x.join()
        x2.join()
        x3.join()
        x4.join()
        # print(df, df.iterators)


if __name__ == '__main__':
    unittest.main()
