

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from towhee.connectors import Connectors


class TestFaiss(unittest.TestCase):
    '''
    Test faiss search
    '''

    def _test_in_memory(self, index_class, size, expect):
        np.random.seed(1234)  # make reproducible
        d = 64
        xb = np.random.random(d).astype('float32')
        with getattr(Connectors, index_class)(d) as index:
            for i in range(size):
                xb = np.random.random((1, d)).astype('float32')
                index.insert(str(i), xb)
            e = np.random.random((1, d)).astype('float32')
            ret = index.search(e, 2)
            for i in range(len(ret)):
                self.assertEqual(ret[i].key, expect[i][0])
                self.assertTrue(abs(ret[i].score - expect[i][1]) < 0.0001)

    def _test_in_disk(self, index_class, size, expect):
        np.random.seed(1234)  # make reproducible
        d = 64
        xb = np.random.random(d).astype('float32')

        with TemporaryDirectory(dir='./') as tp:
            with getattr(Connectors, index_class)(d, tp) as index:
                for i in range(size):
                    xb = np.random.random((1, d)).astype('float32')
                    index.insert(str(i), xb)

            p1 = Path(tp) / 'index.bin'
            self.assertTrue(p1.exists())

            p2 = Path(tp) / 'kv.bin'
            self.assertTrue(p2.exists())

            with getattr(Connectors, index_class)(64, tp) as index:
                e = np.random.random((1, d)).astype('float32')
                ret = index.search(e, 2)
                for i in range(len(ret)):
                    self.assertEqual(ret[i].key, expect[i][0])
                    self.assertTrue(abs(ret[i].score - expect[i][1]) < 0.0001)

    def test_in_memory(self):
        self._test_in_memory('faiss_indexl2', 10000, [('8062', np.float32(6.858701)), ('2699', np.float32(6.956443))])
        self._test_in_memory('hnsw64_index', 1000, [('931', np.float32(5.9835205)), ('50', np.float32(6.1145105))])

    def test_in_disk(self):
        self._test_in_disk('faiss_indexl2', 10000, [('8062', np.float32(6.858701)), ('2699', np.float32(6.956443))])
        self._test_in_disk('hnsw64_index', 1000, [('931', np.float32(5.9835205)), ('50', np.float32(6.1145105))])
