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
import numpy as np

from towhee.array import (
    Array,
    full,
)


class TestArray(unittest.TestCase):
    """Basic test case for `Array`.
    """

    def test_construction_from_list(self):

        def _basic_asserts(array, data):
            self.assertEqual(array.size, len(data))

            for a, b in zip(array, data):
                if isinstance(a, np.ndarray):
                    self.assertFalse((a - b).all())
                else:
                    self.assertEqual(a, b)

            for i, v in enumerate(data):
                if isinstance(array[i], np.ndarray):
                    self.assertFalse((array[i] - v).all())
                else:
                    self.assertEqual(v, array[i])

        data = [1, 2, 3]
        array = Array(data=data)
        _basic_asserts(array, data)

        data = ["a", "b", "c"]
        array = Array(data=data)
        _basic_asserts(array, data)

        data = [3.14]
        array = Array(data=data)
        _basic_asserts(array, data)

        data = [[1, 2], [3, 4]]
        array = Array(data=data)
        _basic_asserts(array, data)

        data = []
        array = Array(data=data)
        _basic_asserts(array, data)

    def test_construction_from_scalar(self):

        def _basic_asserts(array, data):
            self.assertEqual(array.size, 1)
            self.assertEqual(array[0], data)

        data = 1
        array = Array(data=data)
        _basic_asserts(array, data)

        data = "abc"
        array = Array(data=data)
        _basic_asserts(array, data)

        data = 3.14
        array = Array(data=data)
        _basic_asserts(array, data)

    def test_construction_from_ndarray(self):

        def _basic_asserts(array, data):
            if data.size == 1:
                data = data.reshape((1,))

            self.assertEqual(array.size, data.shape[0])

            for a, b in zip(array, data):
                if isinstance(a, np.ndarray):
                    self.assertFalse((a - b).all())
                else:
                    self.assertEqual(a, b)

            for i, v in enumerate(data):
                if isinstance(array[i], np.ndarray):
                    self.assertFalse((array[i] - v).all())
                else:
                    self.assertEqual(v, array[i])

        data = np.ndarray([1, 2, 3])
        array = Array(data=data)
        _basic_asserts(array, data)

        data = np.asarray(None)
        array = Array(data=data)
        _basic_asserts(array, data)

        data = np.array([None])
        array = Array(data=data)
        _basic_asserts(array, data)

        data = np.zeros((2, 3))
        array = Array(data=data)
        _basic_asserts(array, data)

    def test_full(self):

        def _basic_asserts(array, value, n):
            self.assertEqual(array.size, n)

            for i, v in enumerate(array):
                self.assertEqual(v, value)
                self.assertEqual(array[i], value)

        n = 8
        value = 1
        array = full(shape=n, fill_value=value)
        _basic_asserts(array, value, n)

        n = 1
        value = 1
        array = full(shape=n, fill_value=value)
        _basic_asserts(array, value, n)

        n = 0
        value = 1
        array = full(shape=n, fill_value=value)
        _basic_asserts(array, value, n)

        n = 8
        value = "abc"
        array = full(shape=n, fill_value=value)
        _basic_asserts(array, value, n)

        n = 8
        value = None
        array = full(shape=n, fill_value=value)
        _basic_asserts(array, value, n)
