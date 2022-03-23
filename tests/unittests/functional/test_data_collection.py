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
from pathlib import Path
from collections import namedtuple

import towhee.functional.data_collection
import towhee.functional.option
from towhee import ops
from towhee import register
from towhee.functional import DataCollection
from towhee.hparam.hyperparameter import param_scope
from towhee.functional.entity import Entity

public_path = Path(__file__).parent.parent.resolve()


@register(name='myop/add-1')
def add_1(x):
    return x + 1


@register(name='myop/add', output_schema=namedtuple('output', [('result')]))
class MyAdd:
    def __init__(self, val):
        self.val = val

    def __call__(self, x):
        return x + self.val


@register(name='myop/mul')
class MyMul:
    def __init__(self, val):
        self.val = val

    def __call__(self, x):
        return x * self.val


dispatcher = {'add': MyAdd, 'mul': MyMul}


class TestDataCollection(unittest.TestCase):
    """
    tests for data collection
    """
    def test_example_for_basic_api(self):
        dc = DataCollection(range(10))
        result = dc.map(lambda x: x + 1).filter(lambda x: x < 3)
        self.assertListEqual(list(result), [1, 2])

    def test_example_for_dispatch_op(self):
        with param_scope(dispatcher=dispatcher):
            dc = DataCollection(range(5))
            result = dc.add(1)
            self.assertListEqual(list(result), [1, 2, 3, 4, 5])

    def test_example_for_chained_towhee_op(self):
        dc = DataCollection(range(5))
        result = (  #
            dc  #
            >> ops.myop.add(val=1)  #
            >> ops.myop.mul(val=2)  #
        )
        self.assertListEqual(list(result), [2, 4, 6, 8, 10])

    def test_example_for_multiple_line_statement(self):
        dc = DataCollection(range(5))
        result = dc \
            .myop.add(val=1) \
            .myop.mul(val=2) \
            .to_list()
        self.assertListEqual(result, [2, 4, 6, 8, 10])

    def test_fill_entity(self):
        entities = [Entity(num=i) for i in range(5)]
        dc = DataCollection(entities)

        self.assertTrue(hasattr(dc, '_iterable'))
        for i in dc:
            self.assertTrue(hasattr(i, 'num'))
            self.assertTrue(hasattr(i, 'id'))
            self.assertFalse(hasattr(i, 'usage'))

        kvs = {'foo': 'bar'}
        res = dc.fill_entity(usage='test').fill_entity(kvs)

        self.assertTrue(hasattr(res, '_iterable'))
        for i in res:
            self.assertTrue(hasattr(i, 'num'))
            self.assertTrue(hasattr(i, 'id'))
            self.assertEqual(i.usage, 'test')
            self.assertEqual(i.foo, 'bar')

        kvs = {'foo': None}
        res = dc.fill_entity(_ReplaceNoneValue=True, _DefaultKVs=kvs)
        self.assertTrue(hasattr(res, '_iterable'))
        for i in res:
            self.assertTrue(hasattr(i, 'num'))
            self.assertTrue(hasattr(i, 'id'))
            self.assertEqual(i.usage, 'test')
            self.assertEqual(i.foo, 0)

    def test_replace(self):
        entities = [Entity(num=i) for i in range(5)]
        dc = DataCollection(entities)
        j = 0
        for i in dc:
            self.assertTrue(i.num == j)
            j += 1

        dc.replace(num={0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        j = 1
        for i in dc:
            self.assertTrue(i.num == j)
            j += 1

    def test_merge(self):
        entities = [Entity(num=i, cnt=i, total=i, const=i) for i in range(5)]
        dc = DataCollection(entities)

        self.assertTrue(hasattr(dc, '_iterable'))
        for i in dc:
            self.assertTrue(hasattr(i, 'num'))
            self.assertTrue(hasattr(i, 'cnt'))
            self.assertTrue(hasattr(i, 'total'))
            self.assertTrue(hasattr(i, 'const'))
            self.assertEqual(i.num, i.cnt)
            self.assertEqual(i.num, i.total)

        dc.merge('num', 'total', drop_origin=False)
        for i in dc:
            self.assertEqual(i.num, i.cnt)
            self.assertEqual(i.num * 2, i.total)

        dc.merge(['num', 'cnt'], 'total')
        for i in dc:
            self.assertFalse(hasattr(i, 'num'))
            self.assertFalse(hasattr(i, 'cnt'))
            self.assertEqual(i.const * 4, i.total)

    def test_from_json(self):
        json_path = public_path / 'test_util' / 'test_mixins' / 'test.json'
        res = DataCollection.from_json(json_path)

        self.assertTrue(isinstance(res, DataCollection))
        for i in res:
            self.assertTrue(isinstance(i, Entity))

    def test_from_csv(self):
        csv_path = public_path / 'test_util' / 'test_mixins' / 'test.csv'
        res = DataCollection.from_csv(csv_path)

        self.assertTrue(isinstance(res, DataCollection))
        for i in res:
            self.assertTrue(isinstance(i, Entity))


TestDataCollectionExamples = doctest.DocTestSuite(towhee.functional.data_collection)
unittest.TextTestRunner().run(TestDataCollectionExamples)

TestOptionExamples = doctest.DocTestSuite(towhee.functional.option)
unittest.TextTestRunner().run(TestOptionExamples)

if __name__ == '__main__':
    unittest.main()
