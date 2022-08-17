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

from towhee import register
from towhee import DataCollection
from towhee import Entity

import towhee

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


class TestDataCollection(unittest.TestCase):
    """
    tests for data collection
    """

    def test_example_for_basic_api(self):
        dc = towhee.dc(range(10))
        result = dc.map(lambda x: x + 1).filter(lambda x: x < 3)
        self.assertListEqual(list(result), [1, 2])

    def test_example_for_multiple_line_statement(self):
        dc = DataCollection(range(5))
        result = dc \
            .myop.add(val=1) \
            .myop.mul(val=2) \
            .to_list()
        self.assertListEqual(result, [2, 4, 6, 8, 10])

    def test_read_json(self):
        json_path = public_path / 'test_util' / 'test_mixins' / 'test.json'
        res = DataCollection.read_json(json_path)

        self.assertTrue(isinstance(res, DataCollection))
        for i in res:
            self.assertTrue(isinstance(i, Entity))

    def test_from_csv(self):
        csv_path = public_path / 'test_util' / 'test_mixins' / 'test.csv'
        res = DataCollection.read_csv(csv_path)

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

    def test_head(self):
        entities = [Entity(a=i, b=i + 1) for i in range(5)]
        dc = DataCollection(entities)

        dc.head(1)

        json_path = public_path / 'test_util' / 'test_mixins' / 'test.json'
        res = DataCollection.read_json(json_path)

        res.head(1)

    def test_classifier_procedure(self):
        csv_path = public_path / 'test_util' / 'data.csv'
        out = DataCollection.read_csv(csv_path=csv_path).unstream()

        # pylint: disable=unnecessary-lambda
        out = (
            out.runas_op['a', 'a'](func=lambda x: int(x))
                .runas_op['b', 'b'](func=lambda x: int(x))
                .runas_op['c', 'c'](func=lambda x: int(x))
                .runas_op['d', 'd'](func=lambda x: int(x))
                .runas_op['e', 'e'](func=lambda x: int(x))
                .runas_op['target', 'target'](func=lambda x: int(x))
        )

        out = out.tensor_hstack[('a', 'b', 'c', 'd', 'e'), 'fea']()

        train, test = out.split_train_test()

        train.set_training().logistic_regression[('fea', 'target'), 'lr_train_predict'](name='logistic')

        test.set_evaluating(train.get_state()) \
            .logistic_regression[('fea', 'target'), 'lr_evaluate_predict'](name='logistic') \
            .with_metrics(['accuracy', 'recall']) \
            .evaluate['target', 'lr_evaluate_predict']('lr') \
            .report()

        train.set_training().decision_tree[('fea', 'target'), 'dt_train_predict'](name='decision_tree')

        test.set_evaluating(train.get_state()) \
            .decision_tree[('fea', 'target'), 'dt_evaluate_predict'](name='decision_tree') \
            .with_metrics(['accuracy', 'recall']) \
            .evaluate['target', 'dt_evaluate_predict']('dt') \
            .report()

        # train.set_training().svc[('fea', 'target'), 'svm_train_predict'](name='svm_classifier')

        # test.set_evaluating(train.get_state()) \
        #     .svc[('fea', 'target'), 'svm_evaluate_predict'](name='svm_classifier') \
        #     .with_metrics(['accuracy', 'recall']) \
        #     .evaluate['target', 'svm_evaluate_predict']('svm') \
        #     .report()


def load_tests(loader, tests, ignore):
    # pylint: disable=unused-argument
    tests.addTests(doctest.DocTestSuite(towhee.functional.data_collection))
    tests.addTests(doctest.DocTestSuite(towhee.functional.option))
    return tests


if __name__ == '__main__':
    unittest.main()
