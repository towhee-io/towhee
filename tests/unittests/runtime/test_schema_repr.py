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
# WITHOUT_ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from towhee.runtime.schema_repr import SchemaRepr
from towhee.runtime.data_queue import ColumnType


class TestSchemaRepr(unittest.TestCase):
    """
    SchemaRepr test
    """
    def test_repr(self):
        schema = SchemaRepr.from_dag('a', 'map')
        self.assertEqual(schema.name, 'a')
        self.assertEqual(schema.type, ColumnType.SCALAR)

    def test_map_filter(self):
        schema = SchemaRepr.from_dag('a', 'map', [ColumnType.QUEUE])
        self.assertEqual(schema.type, ColumnType.QUEUE)
        schema = SchemaRepr.from_dag('a', 'map', [ColumnType.SCALAR])
        self.assertEqual(schema.type, ColumnType.SCALAR)
        schema = SchemaRepr.from_dag('a', 'map', [ColumnType.SCALAR, ColumnType.SCALAR, ColumnType.QUEUE])
        self.assertEqual(schema.type, ColumnType.QUEUE)

        schema = SchemaRepr.from_dag('a', 'filter', [ColumnType.QUEUE])
        self.assertEqual(schema.type, ColumnType.QUEUE)
        schema = SchemaRepr.from_dag('a', 'filter', [ColumnType.SCALAR])
        self.assertEqual(schema.type, ColumnType.SCALAR)
        schema = SchemaRepr.from_dag('a', 'filter', [ColumnType.SCALAR, ColumnType.QUEUE])
        self.assertEqual(schema.type, ColumnType.QUEUE)

    def test_flat_map_window(self):
        schema = SchemaRepr.from_dag('a', 'flat_map', [ColumnType.QUEUE])
        self.assertEqual(schema.type, ColumnType.QUEUE)
        schema = SchemaRepr.from_dag('a', 'flat_map', [ColumnType.SCALAR])
        self.assertEqual(schema.type, ColumnType.QUEUE)

        schema = SchemaRepr.from_dag('a', 'window', [ColumnType.QUEUE])
        self.assertEqual(schema.type, ColumnType.QUEUE)
        schema = SchemaRepr.from_dag('a', 'window', [ColumnType.SCALAR])
        self.assertEqual(schema.type, ColumnType.QUEUE)

    def test_raise(self):
        with self.assertRaises(ValueError):
            SchemaRepr.from_dag('a', 'flat_maps', [ColumnType.QUEUE])
