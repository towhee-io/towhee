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
from collections import namedtuple

from towhee import ops
from towhee import register
from towhee.functional import DataCollection
from towhee.hparam.hyperparameter import param_scope

import towhee.functional.data_collection
import towhee.functional.option


@register(name='myop/add-1', version='0.1')
def add_1(x):
    return x + 1


@register(name='myop/add',
          version='0.1',
          output_schema=namedtuple('output', [('result')]))
class MyAdd:

    def __init__(self, val):
        self.val = val

    def __call__(self, x):
        return x + self.val


@register(name='myop/mul', version='0.1')
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


TestDataCollectionExamples = doctest.DocTestSuite(
    towhee.functional.data_collection)
unittest.TextTestRunner().run(TestDataCollectionExamples)

TestOptionExamples = doctest.DocTestSuite(towhee.functional.option)
unittest.TextTestRunner().run(TestOptionExamples)

if __name__ == '__main__':
    unittest.main()
