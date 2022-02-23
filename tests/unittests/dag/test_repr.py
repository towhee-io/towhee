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
import copy
from unittest import mock
from collections import namedtuple

from towhee.dag.base_repr import BaseRepr
from towhee.dag.graph_repr import GraphRepr
from towhee.utils.yaml_utils import load_yaml
from tests.unittests.test_util import GRAPH_TEST_YAML, GRAPH_TEST_ISO_DF_YAML, GRAPH_TEST_ISO_OP_YAML, GRAPH_TEST_LOOP_YAML


# @unittest.skip('')
class TestRepr(unittest.TestCase):
    """
    Basic test cases for `Repr`.
    """
    def test_base_repr(self):
        self.assertTrue(BaseRepr.is_valid({'test': 1}, {'test'}))
        self.assertFalse(BaseRepr.is_valid({'test': 1}, {'not_exist'}))

    def _check_graph(self, g_repr):
        # test dataframes generation
        self.assertEqual(len(g_repr.dataframes.values()), 3)
        self.assertTrue('test_df_1' in g_repr.dataframes)
        self.assertTrue('test_df_2' in g_repr.dataframes)
        self.assertEqual(g_repr.dataframes['test_df_1'].columns[0].vtype, 'int')
        # test operators generation
        self.assertEqual(len(g_repr.operators.values()), 2)
        self.assertTrue('test_op_1' in g_repr.operators)
        self.assertTrue('test_op_2' in g_repr.operators)
        # check details in operators
        self.assertEqual(g_repr.operators['test_op_1'].inputs, [{'df': 'test_df_1', 'name': 'k1', 'col': 0}])
        self.assertEqual(g_repr.operators['test_op_2'].inputs, [{'df': 'test_df_2', 'name': 'k1', 'col': 0}])
        self.assertEqual(g_repr.operators['test_op_2'].outputs, [{'df': 'test_df_3'}])
        self.assertEqual(g_repr.operators['test_op_2'].init_args, {'arg1': 1, 'arg2': 'test'})
        self.assertEqual(g_repr.operators['test_op_2'].iter_info, {'type': 'map'})

    def test_graph_repr(self):
        # Check if the inforamtion is loaded properly
        # Load from YAML file
        self._check_graph(GraphRepr.from_yaml(GRAPH_TEST_YAML))
        # Load from string
        with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
            self._check_graph(GraphRepr.from_yaml(f.read()))
        # Load from url
        with mock.patch('requests.get') as mock_get:
            with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
                MockResponse = namedtuple('MockResponse', ['text'])
                mock_get.return_value = MockResponse(f.read())
                self._check_graph(GraphRepr.from_yaml('http://mock.yaml'))

    def test_isolation_or_loop(self):
        # Raise error if given false source information that contain loop or isolation
        self.assertRaises(ValueError, GraphRepr.from_yaml, GRAPH_TEST_ISO_DF_YAML)
        self.assertRaises(ValueError, GraphRepr.from_yaml, GRAPH_TEST_ISO_OP_YAML)
        self.assertRaises(ValueError, GraphRepr.from_yaml, GRAPH_TEST_LOOP_YAML)

        # A proper graph does not contain isolation and loopa
        with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
            data = load_yaml(f)
        graph = GraphRepr.from_dict(data)
        non_iso_df = graph.get_isolated_df()
        self.assertFalse(bool(non_iso_df))
        non_iso_op = graph.get_isolated_op()
        self.assertFalse(bool(non_iso_op))
        none_loop = graph.get_loop()
        self.assertFalse(bool(none_loop))
        # If the graph contains isolated dataframes
        with open(GRAPH_TEST_ISO_DF_YAML, 'r', encoding='utf-8') as f:
            iso_data = load_yaml(f)
        iso_df_graph = GraphRepr.from_dict(iso_data)
        iso_df = iso_df_graph.get_isolated_df()
        self.assertTrue(bool(iso_df))
        # If the graph contains isolated operators
        with open(GRAPH_TEST_ISO_OP_YAML, 'r', encoding='utf-8') as f:
            iso_data = load_yaml(f)
        iso_op_graph = GraphRepr.from_dict(iso_data)
        iso_op = iso_op_graph.get_isolated_op()
        self.assertTrue(bool(iso_op))
        # If the graph contains loops
        with open(GRAPH_TEST_LOOP_YAML, 'r', encoding='utf-8') as f:
            iso_data = load_yaml(f)
        loop_graph = GraphRepr.from_dict(iso_data)
        loop = loop_graph.get_loop()
        self.assertTrue(bool(loop))

    def test_error_yaml(self):
        with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
            data = load_yaml(f)
        err1 = copy.deepcopy(data)
        del err1['dataframes']
        with self.assertRaises(ValueError):
            GraphRepr.from_dict(err1)

        err2 = copy.deepcopy(data)
        del err2['operators'][0]['init_args']
        with self.assertRaises(ValueError):
            GraphRepr.from_dict(err2)

    def test_load_str(self):
        with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
            res = BaseRepr.load_str(f)

        self.assertIsInstance(res, dict)

    def test_load_file(self):
        res = BaseRepr.load_file(GRAPH_TEST_YAML)

        self.assertIsInstance(res, dict)

    def test_load_src(self):
        res_1 = BaseRepr.load_src(GRAPH_TEST_YAML)
        with open(GRAPH_TEST_YAML, 'r', encoding='utf-8') as f:
            res_2 = BaseRepr.load_str(f)

        self.assertEqual(res_1, res_2)

if __name__ == '__main__':
    unittest.main()
