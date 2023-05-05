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
from towhee import DataLoader, pipe, ops, register
from towhee.operator.base import PyOperator


class TestDataLoader(unittest.TestCase):
    """
    TestDataLoader
    """

    def test_iterable(self):
        p = pipe.input('num').map('num', 'ret', lambda x: x + 1).output('ret')
        expect = [2, 3, 4]
        for i, data in enumerate(DataLoader([{'num': 1}, {'num': 2}, {'num': 3}], parser=lambda x: x['num'])):
            self.assertEqual(p(data).to_list()[0][0], expect[i])

        res = []
        for data in DataLoader([{'num': 1}, {'num': 2}, {'num': 3}], parser=lambda x: x['num'], batch_size=2):
            for ret in p.batch(data):
                res.append(ret.to_list()[0][0])
        self.assertEqual(res, expect)

    def test_register_ops(self):
        @register
        class Seq(PyOperator):  # pylint: disable=unused-variable
            def __init__(self, size):
                self._size = size

            def __call__(self):
                yield from range(self._size)

        size = 100
        p = pipe.input('num').map('num', 'ret', lambda x: x + 1).output('ret')
        expect = list(range(1, size + 1))
        for i, data in enumerate(DataLoader(ops.Seq(size))):
            self.assertEqual(p(data).to_list()[0][0], expect[i])

        res = []
        for data in DataLoader(ops.Seq(size), batch_size=7):
            for ret in p.batch(data):
                res.append(ret.to_list()[0][0])
        self.assertEqual(res, expect)

    def test_local_ops(self):
        size = 100
        p = pipe.input('num').map('num', 'ret', lambda x: x + 1).output('ret')
        expect = list(range(1, size + 1))
        for i, data in enumerate(DataLoader(ops.local.data_source_op(size), parser=lambda x: x[1])):
            self.assertEqual(p(data).to_list()[0][0], expect[i])

        res = []
        for data in DataLoader(ops.local.data_source_op(size), parser=lambda x: x[1], batch_size=7):
            for ret in p.batch(data):
                res.append(ret.to_list()[0][0])
        self.assertEqual(res, expect)
