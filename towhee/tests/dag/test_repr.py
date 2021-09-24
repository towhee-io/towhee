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
import yaml
import copy
from unittest import mock
from collections import namedtuple

from towhee.dag.base_repr import BaseRepr
from towhee.dag.graph_repr import GraphRepr

from towhee.tests.test_util import GRAPH_TEST_YAML


class TestRepr(unittest.TestCase):
    """Basic test cases for `hRepr`.
    """

    def test_base_repr(self):
        self.assertTrue(BaseRepr.is_valid({'test': 1}, {'test'}))
        self.assertFalse(BaseRepr.is_valid({'test': 1}, {'not_exist'}))

    def _check_graph(self, g_repr):
        self.assertEqual(len(g_repr.dataframes), 2)
        self.assertEqual(g_repr.dataframes[0].name, 'test_df_1')
        self.assertEqual(g_repr.dataframes[1].name, 'test_df_2')
        self.assertEqual(g_repr.dataframes[0].columns[0].vtype, 'int')

        self.assertEqual(len(g_repr.operators), 2)
        self.assertEqual(g_repr.operators[0].name, 'test_op_1')
        self.assertEqual(g_repr.operators[1].name, 'test_op_2')

        self.assertEqual(g_repr.operators[0].inputs, [
                         {'df': 'test_df_1', 'name': 'k1', 'col': 0}, {'df': 'test_df_2', 'name': 'k1', 'col': 0}])

        self.assertEqual(g_repr.operators[1].inputs, [
                         {'df': 'test_df_1', 'name': 'k1', 'col': 2}])

        self.assertEqual(g_repr.operators[1].outputs, [
                         {'df': 'test_df_1'}])

        self.assertEqual(g_repr.operators[1].init_args,
                         {'arg1': 1, 'arg2': 'test'})

        self.assertEqual(g_repr.operators[1].iter_info,
                         {'type': 'map'})

    def test_graph_repr(self):
        self._check_graph(GraphRepr.from_yaml(GRAPH_TEST_YAML))

        with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
            self._check_graph(GraphRepr.from_yaml(f.read()))

        with mock.patch('requests.get') as mock_get:
            with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
                MockResponse = namedtuple('MockResponse', ['text'])
                mock_get.return_value = MockResponse(f.read())
                self._check_graph(GraphRepr.from_yaml('http://mock.yaml'))

    def test_error_yaml(self):
        with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f.read())
        err1 = copy.deepcopy(data)
        del err1['dataframes']
        with self.assertRaises(ValueError):
            GraphRepr.from_dict(err1)

        err2 = copy.deepcopy(data)
        del err2['operators'][0]['init_args']
        with self.assertRaises(ValueError):
            GraphRepr.from_dict(err2)
