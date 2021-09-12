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
import threading
import time
from typing import Optional

from towhee.dataframe import DataFrame


class DfWriter(threading.Thread):
    """Put data to dataframe
    """

    def __init__(self, df: DataFrame, count: Optional[int] = None):
        super().__init__()
        self._df = df
        self._need_stop = False
        self._count = count
        self._set_sealed = False

    def run(self):
        while not self._need_stop:
            if self._count is not None and self._count <= 0:
                break
            self._df.put(())
            self._count -= 1
            time.sleep(0.05)

        if self._set_sealed:
            self._df.seal()

    def set_sealed_when_stop(self):
        self._set_sealed = True

    def stop(self):
        self.need_stop = True


class TestDataframe(unittest.TestCase):
    """
    Test dataframe basic function
    """

    def test_put(self):
        df = DataFrame('test')
        t = DfWriter(df, 10)
        t.start()
        t.join()
        self.assertEqual(df.name, 'test')
        self.assertEqual(df.size, 10)
        end, datas = df.get(0, 4)
        self.assertFalse(end)
        self.assertEqual(len(datas), 4)

        end, datas = df.get(8, 4)
        self.assertFalse(end)
        self.assertEqual(len(datas), 2)

        end, datas = df.get(8, 4, True)
        self.assertFalse(end)
        self.assertEqual(len(datas), 0)

        self.assertFalse(df.is_sealed())
        df.seal()
        self.assertTrue(df.is_sealed())

        end, datas = df.get(8, 4, True)
        self.assertTrue(end)
        self.assertEqual(len(datas), 2)

    def test_multithread(self):
        df = DataFrame('test')
        t = DfWriter(df, 10)
        t.set_sealed_when_stop()
        t.start()
        index = 0
        while True:
            end, items = df.get(index, 2)
            if not end:
                index += len(items)
            else:
                self.assertEqual(index + len(items), 10)
                break


class TestMapIterator(unittest.TestCase):
    """
    map iterator basic test
    """

    def test_map_iterator_multithread(self):
        df = DataFrame('test')
        it = df.map_iter()
        t = DfWriter(df, 20)
        t.start()
        t.set_sealed_when_stop()

        count = 0
        for item in it:
            if item:
                count += 1
            time.sleep(0.01)
        self.assertEqual(count, 20)

    def test_map_iterator(self):
        df = DataFrame('test')
        it = df.map_iter()
        t = DfWriter(df, 20)
        t.start()
        t.join()

        count = 0
        for item in it:
            if count < 20:
                self.assertEqual(len(item), 1)
                count += 1
                self.assertEqual(it.accessible_size(), 20 - count)
            else:
                df.seal()
