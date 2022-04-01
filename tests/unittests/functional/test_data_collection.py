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
            self.assertFalse(hasattr(i, 'usage'))

        kvs = {'foo': 'bar'}
        res = dc.fill_entity(usage='test').fill_entity(kvs)

        self.assertTrue(hasattr(res, '_iterable'))
        for i in res:
            self.assertTrue(hasattr(i, 'num'))
            self.assertEqual(i.usage, 'test')
            self.assertEqual(i.foo, 'bar')

        kvs = {'foo': None}
        res = dc.fill_entity(_ReplaceNoneValue=True, _DefaultKVs=kvs)
        self.assertTrue(hasattr(res, '_iterable'))
        for i in res:
            self.assertTrue(hasattr(i, 'num'))
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

    def test_runas_op(self):
        def add(x):
            return x + 1

        entities = [Entity(a=i, b=i + 1) for i in range(5)]
        dc = DataCollection(entities)

        res = dc.runas_op['a', 'b'](func=lambda x: x - 1)
        for i in res:
            self.assertTrue(i.a == i.b + 1)

        res = dc.runas_op['a', 'b'](func=add)
        for i in res:
            self.assertTrue(i.a == i.b - 1)

        self.assertRaises(ValueError, dc.runas_op['a', 'b'], add)

    def test_to_dict(self):
        entities = [Entity(a=i, b=i + 1) for i in range(5)]
        dc = DataCollection(entities)

        for i in dc:
            self.assertTrue(isinstance(i.to_dict(), dict))

    def test_head(self):
        entities = [Entity(a=i, b=i + 1) for i in range(5)]
        dc = DataCollection(entities)

        dc.head(1)

        json_path = public_path / 'test_util' / 'test_mixins' / 'test.json'
        res = DataCollection.from_json(json_path)

        res.head(1)

    def test_dropna(self):
        entities = [Entity(a=i, b=i + 1) for i in range(10)]
        entities.append(Entity(a=10, b=''))
        dc = DataCollection(entities)

        for i in dc:
            self.assertTrue(i.b != '' if i.a != 10 else i.b == '')

        res = dc.dropna()
        for i in res:
            self.assertTrue(i.b != '')

    def test_rename(self):
        entities = [Entity(a=i, b=i + 1) for i in range(100000)]
        dc = DataCollection(entities)
        for i in dc:
            self.assertTrue(hasattr(i, 'a') and hasattr(i, 'b'))
            self.assertFalse(hasattr(i, 'A') or hasattr(i, 'B'))

        res = dc.rename(column={'a': 'A', 'b': 'B'})
        for i in res:
            self.assertFalse(hasattr(i, 'a') and hasattr(i, 'b'))
            self.assertTrue(hasattr(i, 'A') or hasattr(i, 'B'))

    def test_classifier_procedure(self):
        csv_path = public_path / 'test_util' / 'data.csv'
        out = DataCollection.from_csv(csv_path=csv_path).unstream()

        # pylint: disable=unnecessary-lambda
        out = (
            out.runas_op['a', 'a'](func=lambda x: int(x))
                .runas_op['b', 'b'](func=lambda x: int(x))
                .runas_op['c', 'c'](func=lambda x: int(x))
                .runas_op['d', 'd'](func=lambda x: int(x))
                .runas_op['e', 'e'](func=lambda x: int(x))
                .runas_op['target', 'target'](func=lambda x: int(x))
        )

        out = out.hstack[('a', 'b', 'c', 'd', 'e'), 'fea']()

        train, test = out.split_train_test()

        train.set_training().logistic_regression[('fea', 'target'), 'lr_train_predict'](name='logistic')

        test.set_evaluating(train.get_state()) \
            .logistic_regression[('fea', 'target'), 'lr_evaluate_predict'](name = 'logistic') \
            .with_metrics(['accuracy', 'recall']) \
            .evaluate['target', 'lr_evaluate_predict']('lr') \
            .report()

        train.set_training().decision_tree[('fea', 'target'), 'dt_train_predict'](name='decision_tree')

        test.set_evaluating(train.get_state()) \
            .decision_tree[('fea', 'target'), 'dt_evaluate_predict'](name = 'decision_tree') \
            .with_metrics(['accuracy', 'recall']) \
            .evaluate['target', 'dt_evaluate_predict']('dt') \
            .report()

        train.set_training().svc[('fea', 'target'), 'svm_train_predict'](name='svm_classifier')

        test.set_evaluating(train.get_state()) \
            .svc[('fea', 'target'), 'svm_evaluate_predict'](name = 'svm_classifier') \
            .with_metrics(['accuracy', 'recall']) \
            .evaluate['target', 'svm_evaluate_predict']('svm') \
            .report()


TestDataCollectionExamples = doctest.DocTestSuite(towhee.functional.data_collection)
unittest.TextTestRunner().run(TestDataCollectionExamples)

TestOptionExamples = doctest.DocTestSuite(towhee.functional.option)
unittest.TextTestRunner().run(TestOptionExamples)

if __name__ == '__main__':
    unittest.main()
