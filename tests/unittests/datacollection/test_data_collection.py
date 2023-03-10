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

import towhee.datacollection.data_collection
from towhee.datacollection import DataCollection
from towhee import pipe


class TestDataCollection(unittest.TestCase):
    """
    Unit test for DataCollection.
    """
    def test_normal(self):
        p = (
            pipe.input('a')
                .map('a', 'b', lambda x: x + 1)
                .output('a', 'b')
        )

        res = p(1)
        dc = DataCollection(res)
        dc.show()

        self.assertEqual(dc[0]['a'], 1)
        self.assertEqual(dc[0]['b'], 2)

    def test_no_schema(self):
        p = (
            pipe.input('a')
                .map('a', 'b', lambda x: x + 1)
                .output()
        )

        res = p(1)
        dc = DataCollection(res)
        dc.show()

        self.assertEqual(len(dc), 0)

    def test_no_data(self):
        p = (
            pipe.input('a')
                .map('a', 'b', lambda x: x + 1)
                .filter(('a', 'b'), ('a', 'b'), 'a', lambda x: x > 5)
                .output('a', 'b')
        )

        res = p(1)
        dc = DataCollection(res)
        dc.show()

        self.assertEqual(len(dc), 0)


def load_tests(loader, tests, ignore):
    # pylint: disable=unused-argument
    tests.addTests(doctest.DocTestSuite(towhee.datacollection.data_collection))
    return tests


if __name__ == '__main__':
    unittest.main()
