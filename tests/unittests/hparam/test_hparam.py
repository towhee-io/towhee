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

import towhee.hparam.hyperparameter
from towhee.hparam import param_scope, HyperParameter


class TestHyperParameter(unittest.TestCase):
    """
    tests for HyperParameter
    """

    def test_parameter_create(self):
        param1 = HyperParameter(a=1, b=2)
        self.assertEqual(param1.a, 1)
        self.assertEqual(param1.b, 2)

        param2 = HyperParameter(**{'a': 1, 'b': 2})
        self.assertEqual(param2.a, 1)
        self.assertEqual(param2.b, 2)

    def test_parameter_update_with_holder(self):
        param1 = HyperParameter()
        param1.a = 1
        param1.b = 2
        param1.c.b.a = 3
        self.assertDictEqual(param1, {'a': 1, 'b': 2, 'c': {'b': {'a': 3}}})

    def test_parameter_update(self):
        param1 = HyperParameter()
        param1.put('c.b.a', 1)
        self.assertDictEqual(param1, {'c': {'b': {'a': 1}}})

    def test_parameter_patch(self):
        param1 = HyperParameter()
        param1.update({'a': 1, 'b': 2})
        self.assertEqual(param1.a, 1)
        self.assertEqual(param1.b, 2)


class TestAccesscor(unittest.TestCase):
    """
    tests for Accesscor
    """

    def test_holder_as_bool(self):
        param1 = HyperParameter()
        self.assertFalse(param1.a.b)

        param1.a.b = False
        self.assertFalse(param1.a.b)

        param1.a.b = True
        self.assertTrue(param1.a.b)


class TestParamScope(unittest.TestCase):
    """
    tests for param_scope
    """

    def test_scope_create(self):
        with param_scope(a=1, b=2) as hp:
            self.assertEqual(hp.a, 1)
            self.assertEqual(hp.b, 2)

        with param_scope(**{'a': 1, 'b': 2}) as hp:
            self.assertEqual(hp.a, 1)
            self.assertEqual(hp.b, 2)

    def test_nested_scope(self):
        with param_scope(a=1, b=2) as hp1:
            self.assertEqual(hp1.a, 1)

            with param_scope(a=3) as hp2:
                self.assertEqual(hp2.a, 3)

    def test_scope_with_function_call(self):

        def read_a():
            with param_scope() as hp:
                return hp.a

        self.assertFalse(read_a())

        with param_scope(a=1):
            self.assertEqual(read_a(), 1)
        with param_scope(a=2):
            self.assertEqual(read_a(), 2)

        with param_scope(a=1):
            self.assertEqual(read_a(), 1)
            with param_scope(a=2):
                self.assertEqual(read_a(), 2)
            self.assertEqual(read_a(), 1)


def load_tests(loader, tests, ignore):
    #pylint: disable=unused-argument
    tests.addTests(doctest.DocTestSuite(towhee.hparam.hyperparameter))
    return tests


if __name__ == '__main__':
    unittest.main()
