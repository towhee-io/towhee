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

import doctest
import unittest
import os
import cv2
import faiss
import numpy as np
from pathlib import Path
from towhee._types.image import Image

import towhee
import towhee.functional.mixins.computer_vision
import towhee.functional.mixins.dataset
import towhee.functional.mixins.display
import towhee.functional.mixins.dataframe
import towhee.functional.mixins.metric
import towhee.functional.mixins.parallel
import towhee.functional.mixins.state
import towhee.functional.mixins.serve
import towhee.functional.mixins.config
import towhee.functional.mixins.remote
import towhee.functional.mixins.data_processing
import towhee.functional.mixins.safe
import towhee.functional.mixins.list
import towhee.functional.mixins.stream

from towhee.functional.mixins.display import _ndarray_brief, to_printable_table
from towhee import DataCollection, DataFrame, dc
from towhee import Entity

public_path = Path(__file__).parent.parent.resolve()


def load_tests(loader, tests, ignore):
    #pylint: disable=unused-argument
    for mod in [
            towhee.functional.mixins.computer_vision,
            towhee.functional.mixins.dataset,
            towhee.functional.mixins.display,
            towhee.functional.mixins.dataframe,
            towhee.functional.mixins.metric,
            towhee.functional.mixins.parallel,
            towhee.functional.mixins.state,
            towhee.functional.mixins.serve,
            towhee.functional.mixins.column,
            towhee.functional.mixins.config,
            towhee.functional.mixins.remote,
            towhee.functional.mixins.list,
            towhee.functional.mixins.data_processing,
            towhee.functional.mixins.stream,
            towhee.functional.mixins.safe,
    ]:
        tests.addTests(doctest.DocTestSuite(mod))

    return tests


# pylint: disable=import-outside-toplevel
class TestSaveMixin(unittest.TestCase):
    """
    Unit test for SaveMixin.
    """
    def test_to_csv(self):
        import pandas as pd
        csv_file = public_path / 'test_util' / 'test_mixins' / 'test.csv'

        record = pd.read_csv(csv_file, encoding='utf-8')
        dc_1 = towhee.from_df(record).unstream()
        dc_2 = DataCollection(((1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7)))
        e = [Entity(a=i, b=i + 1, c=i + 2) for i in range(5)]
        dc_3 = DataFrame(e).unstream()
        dc_4 = DataFrame(iter(e)).unstream()

        out_1 = public_path / 'test_util' / 'test_mixins' / 'test_1.csv'
        out_2 = public_path / 'test_util' / 'test_mixins' / 'test_2.csv'
        out_3 = public_path / 'test_util' / 'test_mixins' / 'test_3.csv'
        out_4 = public_path / 'test_util' / 'test_mixins' / 'test_4.csv'
        dc_1.to_csv(csv_path=out_1)
        dc_2.to_csv(csv_path=out_2)
        dc_3.to_csv(csv_path=out_3)
        dc_4.to_csv(csv_path=out_4)

        self.assertTrue(out_1.is_file())
        self.assertTrue(out_2.is_file())
        self.assertTrue(out_3.is_file())
        self.assertTrue(out_4.is_file())
        out_1.unlink()
        out_2.unlink()
        out_3.unlink()
        out_4.unlink()


class TestMetricMixin(unittest.TestCase):
    """
    Unittest for MetricMixin.
    """
    def test_hit_ratio(self):
        true = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        pred_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        pred_2 = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
        pred_3 = [[0, 11, 12]]

        mhr = towhee.functional.mixins.metric.mean_hit_ratio
        self.assertEqual(1, mhr(true, pred_1))
        self.assertEqual(0.8, mhr(true, pred_2))
        self.assertEqual(0, mhr(true, pred_3))

    def test_average_precision(self):
        true = [[1, 2, 3, 4, 5]]
        pred_1 = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5]]
        pred_2 = [[0, 1, 6, 7, 2, 8, 3, 9, 10]]
        pred_3 = [[0, 11, 12]]

        trues = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        pred_4 = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5], [0, 1, 6, 7, 2, 8, 3, 9, 10]]

        mean_ap = towhee.functional.mixins.metric.mean_average_precision
        self.assertEqual(0.62, round(mean_ap(true, pred_1), 2))
        self.assertEqual(0.44, round(mean_ap(true, pred_2), 2))
        self.assertEqual(0, mean_ap(true, pred_3))
        self.assertEqual(0.53, round(mean_ap(trues, pred_4), 2))


class TestDisplayMixin(unittest.TestCase):
    """
    Unit test for DisplayMixin.
    """
    def test_ndarray_bref(self):
        arr = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
        # pylint: disable=protected-access
        self.assertEqual(_ndarray_brief(arr, 3), '[1.1, 2.2, 3.3, ...] shape=(3, 2)')

    def test_to_printable_table(self):
        dc_1 = DataCollection([[1.1, 2.2], [3.3, 4.4]])
        # pylint: disable=protected-access
        plain_tbl = to_printable_table(dc_1._iterable, tablefmt='plain')
        self.assertEqual(plain_tbl, '1.1  2.2\n3.3  4.4')

        html_tbl = to_printable_table(dc_1._iterable, tablefmt='html')
        html_str = '<table style="border-collapse: collapse;"><tr></tr> '\
                   '<tr><td style="text-align: center; vertical-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3; ">1.1</td> '\
                   '<td style="text-align: center; vertical-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3; ">2.2</td></tr> '\
                   '<tr><td style="text-align: center; vertical-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3; ">3.3</td> '\
                   '<td style="text-align: center; vertical-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3; ">4.4</td></tr></table>'
        self.assertEqual(html_tbl, html_str)

        dc_1 = DataCollection([['hello'], ['world']])
        plain_tbl = to_printable_table(dc_1._iterable, tablefmt='plain')
        self.assertEqual(plain_tbl, 'hello\nworld')

        html_tbl = to_printable_table(dc_1._iterable, tablefmt='html')
        html_str = '<table style="border-collapse: collapse;"><tr></tr> '\
                   '<tr><td style="text-align: center; vertical-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3; ">hello</td></tr> '\
                   '<tr><td style="text-align: center; vertical-align: center; border-right: solid 1px #D3D3D3; '\
                   'border-left: solid 1px #D3D3D3; ">world</td></tr></table>'
        self.assertEqual(html_tbl, html_str)

    def test_show(self):
        logo_path = os.path.join(Path(__file__).parent.parent.parent.parent.resolve(), 'towhee_logo.png')
        img = cv2.imread(logo_path)
        towhee_img = Image(img, 'BGR')
        dc_1 = DataCollection([[1, img, towhee_img], [2, img, towhee_img]])
        dc_1.show(tablefmt='plain')
        dc_1.show(tablefmt='html')


class TestColumnComputing(unittest.TestCase):
    """
    Unit test for column-based computing.
    """
    def test_siso(self):
        df = dc['a'](range(10))\
            .to_column()\
            .runas_op['a', 'b'](func=lambda x: x+1)

        self.assertTrue(all(map(lambda x: x.a == x.b - 1, df)))

    def test_simo(self):
        df = dc['a'](range(10))\
            .to_column()\
            .runas_op['a', ('b', 'c')](func=lambda x: (x+1, x-1))

        self.assertTrue(all(map(lambda x: x.a == x.b - 1, df)))
        self.assertTrue(all(map(lambda x: x.a == x.c + 1, df)))

    def test_miso(self):
        df = dc['a', 'b']([range(10), range(10)])\
            .to_column()\
            .runas_op[('a', 'b'), 'c'](func=lambda x, y: x + y)

        self.assertTrue(all(map(lambda x: x.c == x.a + x.b, df)))

    def test_mimo(self):
        df = dc['a', 'b']([range(10), range(10)])\
            .to_column()\
            .runas_op[('a', 'b'), ('c', 'd')](func=lambda x, y: (x+1, y-1))

        self.assertTrue(all(map(lambda x: x.a == x.c - 1 and x.b == x.d + 1, df)))


class TestFaissMixin(unittest.TestCase):
    """
    Unittest for FaissMixin.
    """
    def test_faiss(self):
        nb = 500
        d = 128
        x = np.random.random((nb, d)).astype('float32')
        x1 = list(v for v in x)
        x2 = x1[0:3]
        ids = list(i for i in range(nb))
        index_path = public_path / 'index.bin'
        if index_path.exists():
            index_path.unlink()

        (
            towhee.dc['id'](ids)
            .runas_op['id', 'vec'](func=lambda x: x1[x])
            .to_faiss['id', 'vec'](findex=str(index_path))
        )
        index = faiss.read_index(str(index_path))
        self.assertTrue(Path(index_path).exists())
        self.assertEqual(index.ntotal, nb)

        res = (
            towhee.dc(x2)
            .faiss_search(findex=str(index_path))
            .to_list()
        )
        self.assertEqual(res[0][0].score, 0)
        self.assertEqual(res[1][0].score, 0)
        self.assertEqual(res[2][0].score, 0)

        index_path.unlink()


class TestCompileMixin(unittest.TestCase):
    """
    Unittest for FaissMixin.
    """
    def test_compile(self):
        import time
        from towhee import register
        @register(name='inner_distance')
        def inner_distance(query, data):
            dists = []
            for vec in data:
                dist = 0
                for i in range(len(vec)):
                    dist += vec[i] * query[i]
                dists.append(dist)
            return dists

        data = [np.random.random((10000, 128)) for _ in range(10)]
        query = np.random.random(128)

        t1 = time.time()
        _ = (
            towhee.dc['a'](data)
                .runas_op['a', 'b'](func=lambda _: query)
                .inner_distance[('b', 'a'), 'c']()
        )
        t2 = time.time()
        _ = (
            towhee.dc['a'](data)
                .config(jit='numba')
                .runas_op['a', 'b'](func=lambda _: query)
                .inner_distance[('b', 'a'), 'c']()
        )
        t3 = time.time()
        self.assertTrue(t3 - t2 < t2 - t1)

    def test_failed_compile(self):
        import time
        from towhee import register
        @register(name='inner_distance1')
        def inner_distance1(query, data):
            data = np.array(data) # numba does not support np.array(data)
            dists = []
            for vec in data:
                dist = 0
                for i in range(len(vec)):

                    dist += vec[i] * query[i]
                dists.append(dist)
            return dists

        data = [np.random.random((10000, 128)) for _ in range(10)]
        query = np.random.random(128)

        t1 = time.time()
        _ = (
            towhee.dc['a'](data)
                .runas_op['a', 'b'](func=lambda _: query)
                .inner_distance1[('b', 'a'), 'c']()
        )
        t2 = time.time()
        _ = (
            towhee.dc['a'](data)
                .config(jit='numba')
                .runas_op['a', 'b'](func=lambda _: query)
                .inner_distance1[('b', 'a'), 'c']()
        )
        t3 = time.time()
        self.assertTrue(t3 - t2 > t2 - t1)

class TestRemoteMixin(unittest.TestCase):
    """
    Unit test for RemoteMixin
    """
    def test_remote(self):
        dc_1 = DataCollection(((1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7)))
        remote = dc_1.remote('127.0.0.1:9001', mode='infer', protocol='grpc')
        self.assertEqual(remote[0], None)

    def test_remote_error(self):
        dc_2 = DataCollection(((1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7)))
        remote = dc_2.remote('127.0.0.1:9000', mode='async_infer', protocol='http')
        self.assertEqual(remote[0], None)

if __name__ == '__main__':
    unittest.main()
