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

from unittest import TestCase

import towhee

# pylint: disable=unused-variable


def add_global(x):
    return x + 1


class AddX:
    """ Mocked add operator
    """

    def __init__(self, x) -> None:
        self._x = x

    def __call__(self, y):
        return y + self._x


@towhee.register
def add_register(x):
    return x + 1


class TestDispatcher(TestCase):
    """ Test cases for missing method dispatcher
    """

    def test_local_function(self):

        def add_local(x):
            return x + 1

        retval = towhee.range(3).add_local().to_list()
        self.assertListEqual(retval, [1, 2, 3])

    def test_local_function_with_schema(self):

        def add_with_schema(x):
            return x + 1

        retval = towhee.dc['a']([0, 1, 2]).add_with_schema['a', 'b']()
        self.assertListEqual(retval.select('b').as_raw().to_list(), [1, 2, 3])

    def test_global_function(self):
        retval = towhee.range(3).add_global().to_list()
        self.assertListEqual(retval, [1, 2, 3])

    def test_global_class(self):
        retval = towhee.range(3).AddX(2).to_list()
        self.assertListEqual(retval, [2, 3, 4])

    # TODO: support import module in towhee dispatcher
    # def test_import_method(self):
    #     retval = towhee.range(3).ops.add_register().to_list()
    #     self.assertListEqual(retval, [1, 2, 3])
