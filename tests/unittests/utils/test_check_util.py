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

from towhee.utils.check_utils import check_set, check_keys, check_node_iter


class TestCheckUtils(unittest.TestCase):
    """
    Unit test for check utils.
    """
    def test_keys(self):
        check_keys({1: 1, 2: 2, 3: 3}, {1, 3})
        with self.assertRaises(ValueError):
            check_keys({1: 1, 2: 2, 3: 3}, {2, 5})

    def test_set(self):
        check_set({1, 2, 3}, {1, 2, 3, 4})
        with self.assertRaises(ValueError):
            check_set({1, 2, 3}, {1, 2, 6, 7})

    def test_node_iter(self):
        check_node_iter('filter', {'filter_columns': 'x'}, ('a', 'b'), ('c', 'd'), {'a', 'b', 'x'})
        with self.assertRaises(ValueError):
            check_node_iter('filter', {'filter_columns': 'y'}, ('a', 'b'), ('c', 'd'), {'a', 'b', 'x'})
        with self.assertRaises(ValueError):
            check_node_iter('filter', {'filter_columns': 'x'}, ('a', 'b'), ('c',), {'a', 'b', 'x'})

        check_node_iter('window', {'size': 3, 'step': 2}, ('a', 'b'), ('c', 'd'), {'a', 'b'})
        with self.assertRaises(ValueError):
            check_node_iter('window', {'size': 'size', 'step': 2}, ('a', 'b'), ('c', 'd'), {'a', 'b'})
        with self.assertRaises(ValueError):
            check_node_iter('window', {'size': 0, 'step': -1}, ('a', 'b'), ('c', 'd'), {'a', 'b'})

        check_node_iter('time_window', {'size': 3, 'step': 2, 'timestamp_col': 'a'}, ('a', 'b'), ('c', 'd'), {'a', 'b'})
        with self.assertRaises(ValueError):
            check_node_iter('time_window', {'size': 3, 'step': 2, 'timestamp_col': 'x'}, ('a', 'b'), ('c', 'd'), {'a', 'b'})
