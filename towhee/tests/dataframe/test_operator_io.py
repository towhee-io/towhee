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
from queue import Queue

from towhee.dataframe import DataFrame, Variable
from towhee.engine._operator_io import MapDataFrameReader

from .dataframe_test_util import DfWriter, MultiThreadRunner


class TestOperatorIO(unittest.TestCase):
    """
    op_ctx IO test
    """

    def test_map_reader(self):
        df = DataFrame('test')
        data = (Variable('int', 1), Variable(
            'str', 'test'), Variable('float', 0.1))
        t = DfWriter(df, 10, data=data)
        t.start()
        t.join()

        map_reader = MapDataFrameReader(df, {'v1': 0, 'v2': 2})
        self.assertEqual(map_reader.size, 10)
        count = 0
        while True:
            item = map_reader.read()
            if not item:
                break
            self.assertEqual(item, {'v1': 1, 'v2': 0.1})
            count += 1
        self.assertEqual(count, 10)
        df.seal()
        self.assertEqual(map_reader.read(), None)

    def test_map_reader_multithread(self):
        df = DataFrame('test')
        data = (Variable('int', 1), Variable(
            'str', 'test'), Variable('float', 0.1))
        data_size = 100
        t = DfWriter(df, data_size, data=data)
        t.set_sealed_when_stop()
        t.start()
        map_reader = MapDataFrameReader(df, {'v1': 0, 'v2': 2})

        q = Queue()

        def read(map_reader: MapDataFrameReader, q: Queue):
            while True:
                item = map_reader.read()
                if item:
                    q.put(item)
                    continue
                elif item is None:
                    break
                else:
                    pass

        runner = MultiThreadRunner(
            target=read, args=(map_reader, q), thread_num=10)

        runner.start()
        runner.join()

        count = 0
        while not q.empty():
            self.assertEqual(q.get(), {'v1': 1, 'v2': 0.1})
            count += 1
        self.assertEqual(count, data_size)
