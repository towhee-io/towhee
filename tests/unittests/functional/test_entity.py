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
from towhee.functional.entity import Entity


class TestEntity(unittest.TestCase):
    """
    Unit test for Entity class.
    """
    def test_init(self):
        d = {'a': 'A', 'b': 'B', 'c': 'C'}
        e1 = Entity()
        e2 = Entity(d)
        e3 = Entity(a='A', b='B', c='C')
        self.assertTrue(isinstance(e1, Entity))
        self.assertTrue(isinstance(e2, Entity))
        self.assertTrue(isinstance(e3, Entity))
        self.assertTrue(isinstance(e1.id, str))
        self.assertEqual(e2.a, e3.a)
        self.assertEqual(e2.b, e3.b)
        self.assertEqual(e2.c, e3.c)
